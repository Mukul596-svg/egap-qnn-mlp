#!/usr/bin/env python3
import asyncio
import csv
import re
import time
import numpy as np
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import itertools
from functools import lru_cache
from aflow import search, K
from mendeleev import element as MendeleevElement
from statistics import stdev
from pymatgen.core import Element as PmgElement
import gc
from joblib import Memory

AURL_PREFIX = "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/"
GEOMETRY_LABELS = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]*)")

# Set up a location for the cache
memory = Memory("element_properties_cache", verbose=0)

# Pre-compute element properties for all common elements
def precompute_element_properties():
    """Build a lookup table of element properties for quick access"""
    elements = {}
    # All elements commonly found in materials
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
    ]
    
    print("Precomputing element properties...")
    for symbol in common_elements:
        try:
            pmg_elem = PmgElement(symbol)
            mnd_elem = MendeleevElement(symbol)
            en = pmg_elem.X
            mendeleev_no = mnd_elem.atomic_number
            group = pmg_elem.group
            
            ec_str = str(mnd_elem.ec)
            max_shell = 0
            valence = 0
            p_electrons = 0
            
            for orbital in ec_str.split():
                shell = int(orbital[0])
                max_shell = max(max_shell, shell)
                
            for orbital in ec_str.split():
                shell = int(orbital[0])
                orbital_type = orbital[1]
                electrons = int(orbital[2:] if len(orbital) > 2 else 1)
                
                if 3 <= group <= 12:  # Transition metals
                    if shell == max_shell and orbital_type == 's':
                        valence += electrons
                    if shell == max_shell - 1 and orbital_type == 'd':
                        valence += electrons
                    if shell == max_shell and orbital_type == 'p':
                        p_electrons = electrons
                else:  # Main group elements
                    if shell == max_shell and orbital_type in ('s', 'p'):
                        valence += electrons
                        if orbital_type == 'p':
                            p_electrons = electrons
                            
            if symbol == 'Y':
                valence = 3
                
            elements[symbol] = {
                'en': en,
                'mendeleev': mendeleev_no,
                'valence': valence,
                'p_electrons': p_electrons,
                'group': group
            }
        except Exception as e:
            print(f"Error precomputing properties for {symbol}: {e}")
    
    print(f"Precomputed properties for {len(elements)} elements")
    return elements

@memory.cache
def get_element_properties(symbol, element_dict):
    """Get element properties from precomputed dict or calculate if missing"""
    if symbol in element_dict:
        return element_dict[symbol]
    
    # Fallback calculation if somehow we missed this element
    print(f"Computing properties for {symbol} on-the-fly (should be rare)")
    pmg_elem = PmgElement(symbol)
    mnd_elem = MendeleevElement(symbol)
    en = pmg_elem.X
    mendeleev_no = mnd_elem.atomic_number
    group = pmg_elem.group
    ec_str = str(mnd_elem.ec)
    max_shell = 0
    valence = 0
    p_electrons = 0
    
    for orbital in ec_str.split():
        shell = int(orbital[0])
        max_shell = max(max_shell, shell)
        
    for orbital in ec_str.split():
        shell = int(orbital[0])
        orbital_type = orbital[1]
        electrons = int(orbital[2:] if len(orbital) > 2 else 1)
        
        if 3 <= group <= 12:  # Transition metals
            if shell == max_shell and orbital_type == 's':
                valence += electrons
            if shell == max_shell - 1 and orbital_type == 'd':
                valence += electrons
            if shell == max_shell and orbital_type == 'p':
                p_electrons = electrons
        else:  # Main group elements
            if shell == max_shell and orbital_type in ('s', 'p'):
                valence += electrons
                if orbital_type == 'p':
                    p_electrons = electrons
                    
    if symbol == 'Y':
        valence = 3
        
    return {
        'en': en,
        'mendeleev': mendeleev_no,
        'valence': valence,
        'p_electrons': p_electrons,
        'group': group
    }

def parse_composition(compound, composition):
    """
    Parse composition and return a sorted tuple of (element, count) tuples.
    Handles different composition formats: list, dict, string, or numpy array.
    """
    try:
        if isinstance(composition, dict):
            # Handle composition as a dictionary
            items = []
            for elem, cnt in composition.items():
                # Ensure count is numeric
                try:
                    count = int(round(float(cnt)))
                except Exception as e:
                    raise ValueError(f"Invalid count for element {elem}: {cnt}") from e
                items.append((elem, count))
            return tuple(sorted(items))
        elif isinstance(composition, list):
            # Handle composition as a list
            symbols = ELEMENT_PATTERN.findall(compound)
            if len(symbols) != len(composition):
                raise ValueError("Mismatch between formula elements and composition counts")
            # Ensure counts are numeric
            try:
                comp_arr = np.array(composition, dtype=int)
            except Exception as e:
                raise ValueError(f"Invalid composition list: {composition}") from e
            gcd = np.gcd.reduce(comp_arr)
            norm_comp = (comp_arr // gcd).tolist()
            return tuple(sorted((symbol, count) for symbol, count in zip(symbols, norm_comp)))
        elif isinstance(composition, str):
            # Handle composition as a string (e.g., "Fe2O3")
            try:
                from pymatgen.core import Composition
                comp = Composition(composition)
                return tuple(sorted(comp.items()))
            except Exception as e:
                raise ValueError(f"Invalid composition string: {composition}") from e
        elif isinstance(composition, np.ndarray):
            # Handle composition as a numpy array
            symbols = ELEMENT_PATTERN.findall(compound)
            if len(symbols) != len(composition):
                raise ValueError("Mismatch between formula elements and composition counts")
            try:
                comp_arr = composition.astype(int)
            except Exception as e:
                raise ValueError(f"Invalid composition numpy array: {composition}") from e
            gcd = np.gcd.reduce(comp_arr)
            norm_comp = (comp_arr // gcd).tolist()
            return tuple(sorted((symbol, count) for symbol, count in zip(symbols, norm_comp)))
        else:
            raise ValueError(f"Unsupported composition format: {type(composition)}")
    except Exception as e:
        raise ValueError(f"Error parsing composition {compound}: {e}")

def calculate_derived_properties(compound, composition, element_dict):
    """Calculate derived properties for a material composition"""
    try:
        counts = parse_composition(compound, composition)
        if not counts:
            return None
        
        # Get properties for all elements using our precomputed dict
        # Convert counts to a list of (element, count) tuples
        props = {}
        syms = []
        for s, _ in counts:
            syms.append(s)
            props[s] = get_element_properties(s, element_dict)
        
        # Calculate electronegativity differences
        diffs = [abs(props[syms[i]]['en'] - props[syms[j]]['en']) 
                for i in range(len(syms)) 
                for j in range(i+1, len(syms))]
        
        avg_en_dev = np.mean(diffs) if diffs else 0
        
        total_atoms = sum(count for _, count in counts)
        avg_mendeleev = sum(props[s]['mendeleev'] * count for s, count in counts) / total_atoms if total_atoms > 0 else 0
        
        total_p = sum(props[s]['p_electrons'] * count for s, count in counts)
        total_val = sum(props[s]['valence'] * count for s, count in counts)
        p_fraction = total_p / total_val if total_val > 0 else 0
        
        # Calculate group standard deviation
        group_list = []
        for s, count in counts:
            group_list.extend([props[s]['group']] * count)
        group_std = stdev(group_list) if len(group_list) > 1 else 0
        
        num_unique_elements = len(counts)
        num_constituent_atoms = total_atoms
        
        return avg_en_dev, avg_mendeleev, p_fraction, group_std, num_unique_elements, num_constituent_atoms
    except Exception as e:
        # Print error without breaking the process
        print(f"Error processing composition {compound}: {e}")
        return None

# Process entry function optimized for multiprocessing
def process_entry(entry_data, element_dict):
    """Process a single entry from AFLOW API data"""
    try:
        raw_aurl = entry_data.get("aurl", None)
        if not raw_aurl or not raw_aurl.startswith(AURL_PREFIX):
            return None
        
        # Extract required properties
        geometry = entry_data.get("geometry", None)
        density = entry_data.get("density", None)
        enthalpy = entry_data.get("enthalpy_formation_atom", None)
        spacegroup = entry_data.get("spacegroup_relax", None)
        volume = entry_data.get("volume_cell", None)
        comp = entry_data.get("compound", None)
        comp_list = entry_data.get("composition", None)
        egap = entry_data.get("Egap", None)
        
        # Validate data
        if any(x is None for x in (geometry, density, enthalpy, spacegroup, volume, comp, comp_list, egap)):
            return None
        
        try:
            egap_val = float(egap)
        except (ValueError, TypeError):
            return None
        
        if egap_val == 0:
            return None

        # Parse composition into a hashable tuple
        try:
            comp_tuple = parse_composition(comp, comp_list)
        except Exception as e:
            print(f"Error parsing composition {comp}: {e}")
            return None
            
        # Create unique key using the composition tuple and spacegroup
        unique_key = (comp_tuple, int(spacegroup))
        
        # Calculate derived properties using the composition tuple
        derived = calculate_derived_properties(comp, comp_list, element_dict)
        if derived is None:
            return None
        
        stripped_aurl = raw_aurl[len(AURL_PREFIX):]
        
        row = [stripped_aurl]
        row.extend(geometry)
        row.extend([
            density,
            enthalpy,
            spacegroup,
            volume,
            f"{derived[0]:.4f}",
            f"{derived[1]:.4f}",
            f"{derived[2]:.4f}",
            f"{derived[3]:.4f}",
            f"{derived[4]:.4f}",
            f"{derived[5]:.4f}",
            f"{egap_val:.4f}"
        ])
        
        return (unique_key, row)
        
    except Exception as e:
        print(f"Error in process_entry: {e}")
        return None

# Extract data directly from Entry objects
async def extract_entry_data(entry, timeout=30):  # Timeout in seconds
    """Extract data from AFLOW Entry object to dict with timeout"""
    try:
        data = {}
        # Extract all relevant properties from the Entry object
        for prop in ["aurl", "geometry", "density", "enthalpy_formation_atom", 
                     "spacegroup_relax", "volume_cell", "compound", "composition", "Egap"]:
            try:
                # Use asyncio.wait_for to set a timeout for each property extraction
                data[prop] = await asyncio.wait_for(asyncio.sleep(0, result=getattr(entry, prop, None)), timeout)
            except asyncio.TimeoutError:
                print(f"Timeout extracting {prop} from entry")
                data[prop] = None
            except Exception as e:
                print(f"Error extracting {prop} from entry: {e}")
                data[prop] = None
        return data
    except Exception as e:
        print(f"Error in extract_entry_data: {e}")
        return None

async def process_batch_async(batch, executor, element_dict, max_concurrent=50):
    """Process a batch of entries asynchronously with improved error handling"""
    # Extract data from Entry objects in parallel
    loop = asyncio.get_event_loop()
    
    # Use asyncio.gather to run extract_entry_data concurrently and get results
    batch_data = await asyncio.gather(
        *[extract_entry_data(entry) for entry in batch]
    )
    
    # Filter out failed extractions
    batch_data = [data for data in batch_data if data is not None]
    
    # Process entry data in ProcessPoolExecutor
    results = []
    
    # Use process pool for CPU-intensive work with enhanced error handling
    with ProcessPoolExecutor(max_workers=max_concurrent) as process_executor:
        futures = [
            loop.run_in_executor(process_executor, process_entry, data, element_dict)
            for data in batch_data
        ]
        
        # Enhanced error handling using as_completed
        batch_results = []
        for future in asyncio.as_completed(futures):
            try:
                result = await future
                if result is not None:
                    batch_results.append(result)
            except Exception as e:
                print(f"Error processing entry in batch: {e}")
        
    # Filter out None results
    results = [r for r in batch_results if r is not None]
    return results

async def fetch_properties_async(n_entries=1200, csv_file="final_set.csv", 
                               batch_size=150, max_concurrent=16, 
                               max_processes=8, buffer_size=1000, max_retries=7):
    """
    Fetch features with optimized resource usage and retry mechanism.
    
    Args:
        n_entries: Maximum number of entries to fetch
        csv_file: Output file path
        batch_size: Optimized to 150 entries per batch for i7 processors
        max_concurrent: Set to 16 to match typical i7 core count
        max_processes: Set to 8 to balance process overhead and parallelism
        buffer_size: Increased to 1000 for better I/O efficiency
        max_retries: Maximum number of retries for a failed batch
    """
    properties = [
        "geometry",
        "density",
        "enthalpy_formation_atom",
        "spacegroup_relax",
        "volume_cell",
        "compound",
        "composition",
        "Egap"
    ]
    
    header = ["aurl"]
    header.extend(GEOMETRY_LABELS)
    header += [
        "density",
        "enthalpy_formation_atom",
        "spacegroup_relax",
        "volume_cell",
        "avg_en_dev",
        "avg_mendeleev", 
        "p_fraction",
        "group_std",
        "num_unique_elements",
        "num_constituent_atoms",
        "egap"
    ]

    # Precompute element properties
    element_dict = precompute_element_properties()
    
    # Create selector with all required properties
    selector = search()
    for prop in properties:
        selector = selector.select(getattr(K, prop))
    
    # Set up CSV writer with buffering
    with open(csv_file, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        
        # Track seen compositions and counters
        seen_compositions = {}
        total_entries = 0
        row_buffer = []
        
        # Set up the iterator
        entry_iterator = selector.__iter__()
        start_time = time.time()
        batch_num = 0
        
        # Process in batches
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            while total_entries < n_entries:
                try:
                    # Add delay between batches to avoid overwhelming the API
                    if batch_num > 0:
                        await asyncio.sleep(0.5)  # Reduced to 0.5 second cooldown between batches
                    
                    batch = list(itertools.islice(entry_iterator, batch_size))
                    if not batch:
                        print("No more entries available from AFLOW API")
                        break
                    
                    batch_num += 1
                    print(f"Processing batch {batch_num} with {len(batch)} entries...")
                    batch_start_time = time.time()
                    
                    # Periodic garbage collection to manage memory
                    if batch_num % 10 == 0:
                        gc.collect()
                    
                    # Process batch with timeout and improved error handling
                    retries = 0
                    while retries < max_retries:
                        try:
                            batch_results = await asyncio.wait_for(
                                process_batch_async(batch, executor, element_dict, max_concurrent),
                                timeout=300  # 5 minute timeout per batch
                            )
                            break  # Break out of retry loop if successful
                        except asyncio.TimeoutError:
                            print(f"Batch {batch_num} timed out, retrying ({retries + 1}/{max_retries})...")
                            retries += 1
                            await asyncio.sleep(120)  # Wait before retrying
                        except Exception as e:
                            print(f"Error processing batch {batch_num}: {e}, retrying ({retries + 1}/{max_retries})...")
                            retries += 1
                            await asyncio.sleep(120)  # Wait before retrying
                    else:
                        print(f"Batch {batch_num} failed after {max_retries} retries, skipping...")
                        continue  # Skip to the next batch

                    # Add unique results to buffer
                    unique_added = 0
                    for unique_key, row in batch_results:
                        if unique_key in seen_compositions:
                            continue
                        
                        seen_compositions[unique_key] = True
                        row_buffer.append(row)
                        total_entries += 1
                        unique_added += 1
                        
                        # Stop if we've reached the target
                        if total_entries >= n_entries:
                            break
                    
                    batch_time = time.time() - batch_start_time
                    print(f"Batch {batch_num} completed in {batch_time:.2f}s: "
                          f"{unique_added} unique entries added")
                    
                    # Write buffer to CSV if it's big enough or we're done
                    if len(row_buffer) >= buffer_size or total_entries >= n_entries:
                        writer.writerows(row_buffer)
                        out_f.flush()
                        print(f"Wrote {len(row_buffer)} rows to CSV")
                        row_buffer = []
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    rate = total_entries / elapsed if elapsed > 0 else 0
                    print(f"Progress: {total_entries}/{n_entries} entries "
                          f"({rate:.2f} entries/sec, elapsed: {elapsed:.1f}s)\n")
                
                except Exception as e:
                    print(f"Unexpected error in batch {batch_num}: {e}")
                    await asyncio.sleep(3)  # Reduced cool down on error
                    continue
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    print(f"Wrote {total_entries} entries to {csv_file}")
    

def fetch_properties_split(n_entries=1200, csv_file="final_set.csv", 
                          batch_size=150, max_concurrent=16, 
                          max_processes=8, buffer_size=1000):
    """
    Main entry point - launches the async event loop with optimized settings for 13th gen i7
    """
    asyncio.run(fetch_properties_async(
        n_entries=n_entries,
        csv_file=csv_file,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        max_processes=max_processes,
        buffer_size=buffer_size
    ))

if __name__ == "__main__":
    fetch_properties_split(
        n_entries=15000,
        csv_file="raw_data.csv",
        batch_size=150,        
        max_concurrent=10,     
        max_processes=8,      
        buffer_size=1000       
    )
