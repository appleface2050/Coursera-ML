# coding:utf-8
import pymysql
db = pymysql.connect(host="60.205.94.60", port=33006, user="andy", password="Bluestacks2017", database="recommend_system")
cursor = db.cursor()

def insert_data_to_mysqldb(data_list):
    if len(data_list) > 39:
        print("data len > 39")
        return
    else:
        while len(data_list) < 39:
            data_list.append("")

        # 补0
        new = []
        for i in data_list[:14]:
            if i == "":
                new.append(0)
            else:
                new.append(i)
        new += data_list[14:]
        data_list = new

        # SQL 插入语句
        sql = """
        INSERT INTO `recommend_system`.`criteo_train`
        (`id`,`label`,`field1`,`field2`,`field3`,`field4`,`field5`,`field6`,`field7`,`field8`,`field9`,`field10`,`field11`,`field12`,`field13`,
        `categorical1`,`categorical2`,`categorical3`,`categorical4`,`categorical5`,`categorical6`,`categorical7`,`categorical8`,`categorical9`,`categorical10`,
        `categorical11`,`categorical12`,`categorical13`,`categorical14`,`categorical15`,`categorical16`,`categorical17`,`categorical18`,`categorical19`,
        `categorical20`,`categorical21`,`categorical22`,`categorical23`,`categorical24`,`categorical25`)
        VALUES ( NULL, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
        '%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s',
        '%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')
        """ % (data_list[0],data_list[1],data_list[2],data_list[3],data_list[4],data_list[5],data_list[6],data_list[7],data_list[8],data_list[9],data_list[10],
        data_list[11],data_list[12],data_list[13],data_list[14],data_list[15],data_list[16],data_list[17],data_list[18],data_list[19],data_list[20],
               data_list[21], data_list[22], data_list[23], data_list[24], data_list[25], data_list[26], data_list[27],
               data_list[28], data_list[29], data_list[30],
               data_list[31], data_list[32], data_list[33], data_list[34], data_list[35], data_list[36], data_list[37],
               data_list[38]
        )

        try:
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            print(sql)
            print (e)
            db.rollback()
            print("insert fail")


if __name__ == '__main__':
    f = open("../data/test.txt", "r")
    count = 10000
    while count >= 0:
        line = f.readline()
        if not line:
            break
        tmp = line.strip('\n')
        # # print(len(line.split('\t')))
        # print(len(line.split('\t')[14:]))
        insert_data_to_mysqldb(tmp.split("\t"))
        count -= 1
        if count % 100 == 0:
            print (count)
    db.close()
