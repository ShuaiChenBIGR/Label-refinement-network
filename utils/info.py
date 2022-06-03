# write history log
def history_log(path, content, write_stat):
    # print(history)
    the_file = open(path, write_stat)
    the_file.write(content)
    the_file.close()