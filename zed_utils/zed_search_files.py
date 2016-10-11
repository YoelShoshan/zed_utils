import os
import fnmatch

#ignore_characters_in_match is a list of characters, for example:  ['.','^']
def list_files_recurse(path, match_substring=None, match_prefix=None, match_suffix=None, case_insensitive=True, ignore_substrings_in_match=None):
    def change_string(str):
        if None == str:
            return str
        if case_insensitive:
            str = str.lower()
        if None != ignore_substrings_in_match:
            for c in ignore_substrings_in_match:
                #str = str.translate({ord(c): ''})
                str = str.replace(c,'')
        return str

    match_substring = change_string(match_substring)
    match_suffix = change_string(match_suffix)
    match_prefix = change_string(match_prefix)

    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filename_for_match = change_string(filename)

            if None != match_prefix:
                if not filename_for_match.startswith(match_prefix):
                    continue

            if None!=match_substring:
                if match_substring not in filename_for_match:
                    continue
            if None!=match_suffix:
                if not filename_for_match.endswith(match_suffix):
                    continue
            matches.append(os.path.join(root, filename))

    return matches

def list_files_directly_in_dir(search_path):
	sub_files = [f for f in os.listdir(search_path) if os.path.isfile(f)]
	return sub_files

def list_subdir_directly_in_dir(search_path):
	sub_dirs = next(os.walk(search_path))[1]
	return sub_dirs
