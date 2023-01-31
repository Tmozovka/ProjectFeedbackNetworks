def append_to_file(file_name, val):
    f = open(file_name, "a")
    f.write("{0}\n".format(str(val)))
    f.close()