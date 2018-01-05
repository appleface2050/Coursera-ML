import datetime
import seaborn as sns
import numpy as np
import pandas as pd
# import MySQLdb as SQL
import pymysql.cursors
import matplotlib.pyplot as plt
from helper import linear_regression as lr
from helper import general as general

con = {"host": '60.205.94.60',
       "port": 33009,
       "user": 'bluestackscn',
       "password": 'Bluestacks2016',
       "db": 'bs_datastats',
       "charset": 'utf8',
       "cursorclass": pymysql.cursors.DictCursor}

con_monitor = {"host": '60.205.94.60',
               "port": 33006,
               "user": 'bluestackscn',
               "password": 'Bluestacks2016',
               "db": 'bst-monitor',
               "charset": 'utf8',
               "cursorclass": pymysql.cursors.DictCursor}


def get_emulator_install_data(start):
    """
    获取某一天的模拟器安装成功人数
    """
    connection = pymysql.connect(**con)
    data = []
    try:
        with connection.cursor() as cursor:
            # 执行sql语句，插入记录
            sql = """
            SELECT result_date, install_success_user FROM stats_emulator
            WHERE result_date ="%s"
            AND scope_id = 1
            """ % (start)
            # print (sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in result:
                data.append(i["install_success_user"])
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                # connection.commit()
    finally:
        connection.close()
    if not data:
        return 0
    else:
        return data[0]



def get_whole_day_result_by_date_range(dt_start, dt_end):
    connection = pymysql.connect(**con)
    data = []
    try:
        with connection.cursor() as cursor:
            # 执行sql语句，插入记录
            sql = """
            SELECT result_date, install_success_user FROM stats_emulator
            WHERE result_date >="%s" and result_date <"%s"
            AND scope_id = 1
            """ % (dt_start, dt_end)
            # print (sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in result:
                data.append(i["install_success_user"])
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                # connection.commit()
    finally:
        connection.close()
    return data


def get_monitor_data_for_one_day_with_hour_sql(start, hour_sql):
    data = [1]
    connection_monitor = pymysql.connect(**con_monitor)
    try:
        with connection_monitor.cursor() as cursor:
            # 执行sql语句，插入记录
            sql = """
          SELECT HOUR, install_success_user FROM monitor_odps_emulatormonitorstats
          WHERE result_date="%s" AND HOUR in (%s)
          ORDER BY HOUR asc
            """ % (start, hour_sql)
            # print (sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in result:
                data.append(i["install_success_user"])
    finally:
        connection_monitor.close()
    return data


def get_monitor_data_for_one_day(dt_start):
    data = [1]
    connection_monitor = pymysql.connect(**con_monitor)
    try:
        with connection_monitor.cursor() as cursor:
            # 执行sql语句，插入记录
            sql = """
          SELECT HOUR, install_success_user FROM monitor_odps_emulatormonitorstats
          WHERE result_date="%s" AND HOUR in ("00", "01", "02", "03", "04", "05", "06", "07", "08")
          ORDER BY HOUR asc
            """ % (dt_start)
            # print (sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in result:
                data.append(i["install_success_user"])
                # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                # connection_monitor.commit()
    finally:
        # pass
        connection_monitor.close()
    return data


def get_monitor_data_by_date_range_and_hour(hour, dt_start, dt_end):
    connection_monitor = pymysql.connect(**con_monitor)
    data = []
    try:
        with connection_monitor.cursor() as cursor:
            # 执行sql语句，插入记录
            sql = """
          SELECT result_date, HOUR, install_success_user FROM monitor_odps_emulatormonitorstats
          WHERE result_date >="%s" AND result_date< "%s" AND HOUR = "%s"
          ORDER BY result_date asc
            """ % (dt_start, dt_end, hour)
            # print (sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in result:
                data.append(i["install_success_user"])
                # # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
                # connection.commit()
    finally:
        connection_monitor.close()
    return data


def predict_for_one_day(start):
    """
    查看一天的预测结果和预测效果
    """
    data_list = get_monitor_data_for_one_day(start)
    x = np.array(data_list)
    y_predict = x @ theta_ne
    # y_real = get_emulator_install_data(start)
    # print("predict:", y_predict, "误差", abs(y_real-y_predict)/float(y_real))
    return int(y_predict)


def get_next_90_day(start):
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    start_90_before = start - datetime.timedelta(days=91)
    return start_90_before.strftime('%Y-%m-%d')


def predict_for_one_day_all_hours(start, hours=None):
    """
    生成一天所有小时的预测数据， 用这天之前90日数据来计算
    """
    # init_monitor_dict = {}
    if hours is None:
        hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
                 "18", "19", "20", "21", "22"]
    hour_in_use_list = []
    for hour in hours:
        init_monitor_dict = {}
        hour_in_use_list.append(hour)
        # print(hour_in_use_list)
        hour_sql = ""
        for h in hour_in_use_list:
            item = "'%s'," % str(h)
            hour_sql += item
        hour_sql = hour_sql.strip(",")
        # print(hour_sql)
        # sql = """
        #   SELECT HOUR, install_success_user FROM monitor_odps_emulatormonitorstats
        #   WHERE result_date="%s" AND HOUR in (%s)
        #   ORDER BY HOUR asc
        # """ % (start, hour_sql)
        # print (sql)


        # generate X
        if hour >= "08":
            for h in hour_in_use_list:
                init_monitor_dict[h] = get_monitor_data_by_date_range_and_hour(h, get_next_90_day(start), start)
                # print(init_monitor_dict)
            init_monitor_dict["y"] = get_whole_day_result_by_date_range(get_next_90_day(start), start)
            df_monitor = pd.DataFrame(init_monitor_dict)
            # print(df_monitor)
            ones = pd.DataFrame({'ones': np.ones(len(df_monitor))})
            data = pd.concat([ones, df_monitor], axis=1)  # column concat
            X = data.iloc[:, :-1].as_matrix()  # this return ndarray, not matrix
            y = np.array(df_monitor.iloc[:, -1])
            theta_ne = lr.normal_equations(X, y)
            # print(theta_ne)

            data_list = get_monitor_data_for_one_day_with_hour_sql(start, hour_sql)
            x = np.array(data_list)
            y_predict = x @ theta_ne
            y_real = get_emulator_install_data(start)
            # print("predict:", y_predict, "误差", abs(y_real-y_predict)/float(y_real))
            try:
                error_rate = abs(y_real - y_predict) / float(y_real)
            except Exception as e:
                error_rate = 0
            print("start:", start, "hour:", hour, "predict:",int(y_predict), "real:",y_real, "error rate:",)


def predict_for_one_day_with_p(start):
    """
    查看一天的预测结果和预测效果
    """
    data_list = get_monitor_data_for_one_day(start)
    x = np.array(data_list)
    y_predict = x @ theta_ne
    y_real = get_emulator_install_data(start)
    # print("predict:", y_predict, "误差", abs(y_real-y_predict)/float(y_real))
    return (int(y_predict), y_real, abs(y_real - y_predict) / float(y_real))


# init_monitor_dict = {}

# try:
#     with connection.cursor() as cursor:
#         # 执行sql语句，插入记录
#         sql_date = 'select DISTINCT DATE_FORMAT(result_date, "%Y-%m-%d") as dt from stats_emulator where scope_id=1 and result_date >="2017-09-01" and result_date < "2017-09-30"'
#         cursor.execute(sql_date)
#         result = cursor.fetchall()
#         for i in result:
#             # print(i)
#             dt_list.append(i["dt"])
#     # 没有设置默认自动提交，需要主动提交，以保存所执行的语句
#     connection.commit()
# finally:
#     pass

# print (dt_list)
# init_monitor_dict["date"] = dt_list

# for hour in ["00", "01", "02", "03", "04", "05", "06", "07", "08"]:
#     init_monitor_dict[hour] = get_monitor_data_by_date_range_and_hour(hour, "2017-10-01", "2017-12-10")
#
# init_monitor_dict["y"] = get_whole_day_result_by_date_range("2017-10-01", "2017-12-10")
#
# # print (init_monitor_dict)
#
# df_monitor = pd.DataFrame(init_monitor_dict)
# print(df_monitor)
# print(df_monitor.info())

# connection.close()
# connection_monitor.close()

# df_show = pd.DataFrame({"x":df_monitor["05"], "y":df_monitor["y"]})

# sns.lmplot("01","y", df_monitor)
# # plt.plot(df_monitor["00"], df_monitor["y"])
# plt.show()

# get X


# ones = pd.DataFrame({'ones': np.ones(len(df_monitor))})
# data = pd.concat([ones, df_monitor], axis=1)  # column concat
# X = data.iloc[:, :-1].as_matrix()  # this return ndarray, not matrix
# y = np.array(df_monitor.iloc[:, -1])

# alpha = 0.01
# theta = np.zeros(X.shape[1])
# theta.shape
#
# epoch = 500
# final_theta, cost_data = lr.batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
# cost_data[-1]

# theta_ne = lr.normal_equations(X, y)
# print(theta_ne)
# print(lr.cost(theta_ne, X, y))

###################查看一日的多时间点，预测准确率
# print(predict_for_one_day_with_p("2017-12-10"))
# print(predict_for_one_day("2017-12-11"))

predict_for_one_day_all_hours("2017-12-12", ["00","01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])





