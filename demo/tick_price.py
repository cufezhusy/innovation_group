class market_price_evaluator_service(object):
    def __init__(self, file, sym):
        self.file = file
        self.sym = sym
        self.temp_list = None

    def gen_raw_ot_tick(self):
        file = "output.txt" # generated from cli, requires OT installed
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        nfile = "tick_data.txt"
        tmp_list = []
        for i in range(0, len(content)):
            if content[i][:15] == "Processing tick":
                tmp_list.append([content[i][17:35], content[i + 1]])
        tmp_file = open(nfile, 'w')
        for i in tmp_list:
            tmp_file.write("%s\n" % i)
        tmp_file.close()

    def preprocess_ontetick(self):
        with open(self.file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content

    def get_ticker(self):
        return self.sym

    def get_tick(self, tick):
        if self.temp_list is None:
            self.pre_load()
        return self.temp_list[tick][0], float(self.temp_list[tick][1])

    def pre_load(self):
        content = self.preprocess_ontetick()
        tmp_list = []
        for i in range(0, len(content)):
            tmp_list.append(content[i].split(","))
        self.temp_list = tmp_list

if __name__ == "__main__":
    import time
    mpes_obj = market_price_evaluator_service(file="tick_data.txt", sym="EUR=")
    mpes_obj.pre_load()
    for s in range(len(mpes_obj.temp_list)):
        time.sleep(0.1)
        print(mpes_obj.get_tick(s)[1])
