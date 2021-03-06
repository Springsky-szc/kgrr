import xlrd
import csv
import pickle

def read_from_pickle(fname):
    with open(fname, 'rb') as f:
        n = pickle.load(f)
        result = []
        for i in range(n):
            result.append(pickle.load(f))
    return result

def list2dict(lists, i, j):
    dicts = {}
    for line in lists:
        dicts[line[i]] = line[j]
    return dicts

def read_xls(fname):
    #read data from excel to list
    book = xlrd.open_workbook(fname)
    table = book.sheet_by_name('Sheet1')
    raw_number = table.nrows
    lists = []
    for i in range(raw_number):
        lists.append(table.row_values(i))
    return lists

def construct_comments():
    comments = []
    with open('../data/douban/original/cmmt.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            user, records = ''.join(list(filter(str.isdigit,row[0]))), row[1:]
            for record in records:
                record = record.replace('{', '').replace('"', '').replace(':', '#').split('#')
                if(len(record) >= 3 and record[0].isnumeric() and record[1] in {'1', '2', '3', '4', '5'}):
                    movie, rate, time, label = int(id_dict[float(record[0])] - 1), int(record[1]), int(record[2].replace('-','')), record[3].split(' ')
                    label = label[1:] if(label[0] == '标签') else []
                    comments.append([user, movie, rate, time, label])
        user_list = [str(line[0]) for line in comments]
        user_id_dict, _ = construct_id_dict(user_list)
        for i in range(len(comments)):
            comments[i][0] = user_id_dict[comments[i][0]]
    return  comments

def construct_id_dict(lists):
    id = 0
    dicts = {}
    for v in lists:
        if v not in dicts:
            dicts[v] = id
            id += 1
    return dicts, id

def save_pickle(data_list, pickle_file):
    with open(pickle_file, 'wb') as f:
        for data in [len(data_list)] + data_list:
            pickle.dump(data, f)

def construct_attr(attr):
    data_table = read_xls('../data/douban/original/number_' + attr + '.xls')
    data_table = [[int(line[0]) - 1, [data for data in line[1].replace(' ', '').split('/') if data not in {'None','???'}]] for line in data_table]
    data_list = [data for line in data_table for data in line[1]]
    data_dict, _ = construct_id_dict(data_list)
    data_table = [[line[0], [data_dict[data] for data in line[1]]] for line in data_table]
    save_pickle([data_dict, data_table], attr)

if __name__ == '__main__':
    rate_list = read_xls('../data/douban/original/number_rate.xls') #[1.0, 8.2, 218.0]
    id_list = read_xls('../data/douban/original/number_id.xls') #[1,3542816]
    id_dict = list2dict(id_list, 1, 0) #{10548265.0: 41783.0} 电影原始编号：新id+1
    save_pickle([id_list, id_dict], 'id')
    comments = construct_comments() #['78835', 1418752, 4, 20170315, ['宗教', '心理', '悬疑', '惊悚', '意外结局', '反转']]
    save_pickle([comments], 'comments')
    for attr in {'actors', 'type', 'directors', 'writers'}:
        construct_attr(attr)