


def find_duplicates_in_text_file(filename):
    f = open(filename,'r')
    d = {}
    for i,l in enumerate(f.readlines()):
        if l in d:
            print('Error! {0} already found at line {1}!'.format(l,i))
        d[l] = i
    f.close()