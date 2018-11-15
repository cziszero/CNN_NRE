import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fi:
    with open('pretty' + sys.argv[1], 'w', encoding='utf-8') as fo:
        for i in fi:
            if i.find('\x08') == -1:
                fo.write(i)
