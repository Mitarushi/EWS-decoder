import datetime
import numpy as np


class EWSDecoder:
    # 実はbig endianらしい
    # https://web.archive.org/web/20220115184027/https://www.tele.soumu.go.jp/horei/reiki_honbun/a721030001.html

    # 前置符号
    PRE_START_CODE = "1100"
    PRE_END_CODE = "0011"
    # 固定符号
    STATIC_PRIMARY_START_CODE = "0000111001101101"
    STATIC_SECONDARY_START_CODE = "1111000110010010"
    # 地域符号
    REGION_CODES = {
        # 地域共通符号
        "001101001101": "地域共通",
        # 広域符号
        "010110100101": "関東広域圏",
        "011100101010": "中京広域圏",
        "100011010101": "近畿広域圏",
        "011010011001": "鳥取・島根圏",
        "010101010011": "岡山・香川圏",
        # 県域符号
        "000101101011": "北海道",
        "010001100111": "青森県",
        "010111010100": "岩手県",
        "011101011000": "宮城県",
        "101011000110": "秋田県",
        "111001001100": "山形県",
        "000110101110": "福島県",
        "110001101001": "茨城県",
        "111000111000": "栃木県",
        "100110001011": "群馬県",
        "011001001011": "埼玉県",
        "000111000111": "千葉県",
        "101010101100": "東京都",
        "010101101100": "神奈川県",
        "010011001110": "新潟県",
        "010100111001": "富山県",
        "011010100110": "石川県",
        "100100101101": "福井県",
        "110101001010": "山梨県",
        "100111010010": "長野県",
        "101001100101": "岐阜県",
        "101001011010": "静岡県",
        "100101100110": "愛知県",
        "001011011100": "三重県",
        "110011100100": "滋賀県",
        "010110011010": "京都府",
        "110010110010": "大阪府",
        "011001110100": "兵庫県",
        "101010010011": "奈良県",
        "001110010110": "和歌山県",
        "110100100011": "鳥取県",
        "001100011011": "島根県",
        "001010110101": "岡山県",
        "101100110001": "広島県",
        "101110011000": "山口県",
        "111001100010": "徳島県",
        "100110110100": "香川県",
        "000110011101": "愛媛県",
        "001011100011": "高知県",
        "011000101101": "福岡県",
        "100101011001": "佐賀県",
        "101000101011": "長崎県",
        "100010100111": "熊本県",
        "110010001101": "大分県",
        "110100011100": "宮崎県",
        "110101000101": "鹿児島県",
        "001101110010": "沖縄県",
    }
    # 日符号
    DAY_CODES = {
        "10000": "1日",
        "01000": "2日",
        "11000": "3日",
        "00100": "4日",
        "10100": "5日",
        "01100": "6日",
        "11100": "7日",
        "00010": "8日",
        "10010": "9日",
        "01010": "10日",
        "11010": "11日",
        "00110": "12日",
        "10110": "13日",
        "01110": "14日",
        "11110": "15日",
        "00001": "16日",
        "10001": "17日",
        "01001": "18日",
        "11001": "19日",
        "00101": "20日",
        "10101": "21日",
        "01101": "22日",
        "11101": "23日",
        "00011": "24日",
        "10011": "25日",
        "01011": "26日",
        "11011": "27日",
        "00111": "28日",
        "10111": "29日",
        "01111": "30日",
        "11111": "31日",
    }
    # 月符号
    MONTH_CODES = {
        "10001": "1月",
        "01001": "2月",
        "11001": "3月",
        "00101": "4月",
        "10101": "5月",
        "01101": "6月",
        "11101": "7月",
        "00011": "8月",
        "10011": "9月",
        "01011": "10月",
        "11011": "11月",
        "00111": "12月",
    }
    # 時符号
    TIME_CODES = {
        "00011": "0時",
        "10011": "1時",
        "01011": "2時",
        "11011": "3時",
        "00111": "4時",
        "10111": "5時",
        "01111": "6時",
        "11111": "7時",
        "00001": "8時",
        "10001": "9時",
        "01001": "10時",
        "11001": "11時",
        "00101": "12時",
        "10101": "13時",
        "01101": "14時",
        "11101": "15時",
        "00010": "16時",
        "10010": "17時",
        "01010": "18時",
        "11010": "19時",
        "00110": "20時",
        "10110": "21時",
        "01110": "22時",
        "11110": "23時",
    }

    @staticmethod
    def seireki_to_showa(year):
        return year - 1925

    LATEST_YEAR = datetime.datetime.now().year

    # 年符号
    # 年符号は昭和%10で決まる
    MOD10_YEAR_CODES = [
        "10101",
        "01101",
        "11101",
        "00011",
        "10011",
        "01011",
        "10001",
        "01001",
        "11001",
        "00101",
    ]
    YEAR_CODES = (
        lambda MOD10_YEAR_CODES, seireki_to_showa, LATEST_YEAR: {
            MOD10_YEAR_CODES[seireki_to_showa(year) % 10]: f"{year}年"
            for year in range(LATEST_YEAR, LATEST_YEAR - 10, -1)
        }
    )(MOD10_YEAR_CODES, seireki_to_showa, LATEST_YEAR)

    RESULT_BLOCK_END = "BLOCK_END"

    @staticmethod
    def receive_n_bits(first_yield, n):
        results = []
        for _ in range(n):
            received_bit = yield first_yield
            assert received_bit in ("0", "1")
            results.append(received_bit)
            first_yield = None
        return "".join(results)

    @classmethod
    def decode(cls):
        last_output = None

        # 前置符号
        pre_code = yield from cls.receive_n_bits(last_output, 4)
        if pre_code == cls.PRE_START_CODE:
            last_output = "第1/2種開始信号"
        elif pre_code == cls.PRE_END_CODE:
            last_output = "終了信号"
        else:
            raise ValueError(f"Unexpected pre_code: {pre_code}")
        is_end = pre_code == cls.PRE_END_CODE

        while True:
            # 固定符号
            static_code = yield from cls.receive_n_bits(last_output, 16)
            if static_code == cls.STATIC_PRIMARY_START_CODE:
                last_output = "第1種開始信号/終了信号"
            elif static_code == cls.STATIC_SECONDARY_START_CODE:
                last_output = "第2種開始信号"
            else:
                raise ValueError(f"Unexpected static_code: {static_code}")

            # 地域符号
            region_code_head = yield from cls.receive_n_bits(last_output, 2)
            if region_code_head != ("01" if is_end else "10"):
                raise ValueError(f"Unexpected region_code_head: {region_code_head}")
            last_output = ""

            region_code = yield from cls.receive_n_bits(last_output, 12)
            if region_code not in cls.REGION_CODES:
                raise ValueError(f"Unknown region code: {region_code}")
            last_output = cls.REGION_CODES[region_code]

            region_code_tail = yield from cls.receive_n_bits(last_output, 2)
            if region_code_tail != ("11" if is_end else "00"):
                raise ValueError(f"Unexpected region_code_tail: {region_code_tail}")
            last_output = ""

            # 固定符号2
            static_code2 = yield from cls.receive_n_bits(last_output, 16)
            if static_code != static_code2:
                raise ValueError(
                    f"Static codes do not match: {static_code} != {static_code2}"
                )
            if static_code2 == cls.STATIC_PRIMARY_START_CODE:
                last_output = "第1種開始信号/終了信号"
            elif static_code2 == cls.STATIC_SECONDARY_START_CODE:
                last_output = "第2種開始信号"
            else:
                raise ValueError(f"Unexpected static_code2: {static_code2}")

            # 月日区分符号
            month_day_code_head = yield from cls.receive_n_bits(last_output, 3)
            if month_day_code_head != ("100" if is_end else "010"):
                raise ValueError(
                    f"Unexpected month_day_code_head: {month_day_code_head}"
                )
            last_output = ""

            day_code = yield from cls.receive_n_bits(last_output, 5)
            if day_code not in cls.DAY_CODES:
                raise ValueError(f"Unknown day code: {day_code}")
            last_output = cls.DAY_CODES[day_code]

            month_date_code_sep = yield from cls.receive_n_bits(last_output, 1)
            if month_date_code_sep == "0":
                last_output = "今日"
            elif month_date_code_sep == "1":
                last_output = "その他"

            month_code = yield from cls.receive_n_bits(last_output, 5)
            if month_code not in cls.MONTH_CODES:
                raise ValueError(f"Unknown month code: {month_code}")
            last_output = cls.MONTH_CODES[month_code]

            month_day_code_tail = yield from cls.receive_n_bits(last_output, 2)
            if month_day_code_tail != ("11" if is_end else "00"):
                raise ValueError(
                    f"Unexpected month_day_code_tail: {month_day_code_tail}"
                )
            last_output = ""

            # 固定符号3
            static_code3 = yield from cls.receive_n_bits(last_output, 16)
            if static_code3 != static_code2:
                raise ValueError(
                    f"Static codes do not match: {static_code2} != {static_code3}"
                )
            if static_code3 == cls.STATIC_PRIMARY_START_CODE:
                last_output = "第1種開始信号/終了信号"
            elif static_code3 == cls.STATIC_SECONDARY_START_CODE:
                last_output = "第2種開始信号"
            else:
                raise ValueError(f"Unexpected static_code3: {static_code3}")

            # 年時区分符号
            year_time_code_head = yield from cls.receive_n_bits(last_output, 3)
            if year_time_code_head != ("101" if is_end else "011"):
                raise ValueError(
                    f"Unexpected year_time_code_head: {year_time_code_head}"
                )
            last_output = ""

            time_code = yield from cls.receive_n_bits(last_output, 5)
            if time_code not in cls.TIME_CODES:
                raise ValueError(f"Unknown time code: {time_code}")
            last_output = cls.TIME_CODES[time_code]

            year_time_code_sep = yield from cls.receive_n_bits(last_output, 1)
            if year_time_code_sep == "0":
                last_output = "今年"
            elif year_time_code_sep == "1":
                last_output = "その他"

            year_code = yield from cls.receive_n_bits(last_output, 5)
            if year_code not in cls.YEAR_CODES:
                raise ValueError(f"Unknown year code: {year_code}")
            last_output = cls.YEAR_CODES[year_code]

            year_time_code_tail = yield from cls.receive_n_bits(last_output, 2)
            if year_time_code_tail != ("11" if is_end else "00"):
                raise ValueError(
                    f"Unexpected year_time_code_tail: {year_time_code_tail}"
                )
            last_output = cls.RESULT_BLOCK_END


class FSK:
    def __init__(
        self,
        freq_points,
        signal_freq_list,
        delta_hertz,
        sample_per_bit,
        accept_freq_diff,
        peak_width_ratio,
        signal_noise_threshold,
        required_signal_ratio,
    ):
        self.freq_points = freq_points
        self.freq_list = signal_freq_list
        self.delta_hertz = delta_hertz
        self.sample_per_bit = sample_per_bit

        self.accept_freq_diff = accept_freq_diff
        self.peak_width_ratio = peak_width_ratio
        self.signal_noise_threshold = signal_noise_threshold
        self.required_signal_len = int(round(sample_per_bit * required_signal_ratio))

        self.peak_mask = self.get_peak_mask()

        self.update_count = 0

        self.history = [None] * self.sample_per_bit

        self.lock_time = None

        self.decoder = None

    def get_peak_mask(self):
        peak_mask = np.ones(self.freq_points, dtype=bool)
        for freq in self.freq_list:
            # 検出したいピークの裾野は無視する
            min_freq = max(0, self.freq_to_point(freq * (1 - self.peak_width_ratio)))
            max_freq = min(
                self.freq_points - 1,
                self.freq_to_point(freq * (1 + self.peak_width_ratio)),
            )
            peak_mask[min_freq : max_freq + 1] = False

            narrow_min_freq = max(0, self.freq_to_point(freq - self.accept_freq_diff))
            narrow_max_freq = min(
                self.freq_points - 1, self.freq_to_point(freq + self.accept_freq_diff)
            )
            peak_mask[narrow_min_freq : narrow_max_freq + 1] = True
        return peak_mask

    def get_peak_top_k(self, freq_data, k=1):
        is_peak = (
            (freq_data > np.roll(freq_data, 1))
            & (freq_data >= np.roll(freq_data, -1))
            & self.peak_mask
        )
        peak_indices = np.where(is_peak)[0]
        peak_values = freq_data[peak_indices]
        top_k_indices = np.argsort(peak_values)[-k:][::-1]
        return peak_indices[top_k_indices]

    def freq_to_point(self, freq):
        return int(round(freq / self.delta_hertz))

    def point_to_freq(self, point):
        return point * self.delta_hertz

    def find_signal(self, freq_data):
        k = len(self.freq_list) + 1
        peak_indices = self.get_peak_top_k(freq_data, k)
        peak_amplitudes = freq_data[peak_indices]
        peak_freqs = self.point_to_freq(peak_indices)

        detected_signals = []
        detected_signal_amplitudes = []
        non_signal_peak = None
        for peak_freq, peak_amplitude in zip(peak_freqs, peak_amplitudes):
            signal_idx = -1
            for idx, signal_freq in enumerate(self.freq_list):
                if abs(peak_freq - signal_freq) <= self.accept_freq_diff:
                    signal_idx = idx
                    break
            if signal_idx == -1 or signal_idx in detected_signals:
                if non_signal_peak is None:
                    non_signal_peak = peak_amplitude
            else:
                detected_signals.append(signal_idx)
                detected_signal_amplitudes.append(peak_amplitude)

        assert non_signal_peak is not None

        threshold = non_signal_peak * self.signal_noise_threshold
        filtered_signals = [
            signal_idx
            for signal_idx, amplitude in zip(
                detected_signals, detected_signal_amplitudes
            )
            if amplitude >= threshold
        ]

        return filtered_signals

    def update(self, freq_data):
        self.update_count += 1

        signals = self.find_signal(freq_data)

        self.history.pop(0)
        if signals:
            self.history.append(signals[0])
        else:
            self.history.append(None)

        signal_counts = [
            self.history.count(idx) for idx, _ in enumerate(self.freq_list)
        ]
        max_count = max(signal_counts)

        if self.lock_time is None:
            if max_count >= self.required_signal_len:
                self.lock_time = self.update_count - self.required_signal_len

                self.decoder = EWSDecoder.decode()
                self.decoder.__next__()

                print(f"Signal locked at {self.update_count}", flush=True)
        elif (self.update_count - self.lock_time) % self.sample_per_bit == 0:
            if max_count < self.required_signal_len:
                duration = (
                    self.update_count - self.lock_time
                ) // self.sample_per_bit - 1
                print(f"\nSignal lost after {duration} bits\n", flush=True)
                self.lock_time = None
            else:
                bit = str(signal_counts.index(max_count))
                print(bit, end="", flush=True)
                try:
                    res = self.decoder.send(bit)
                    if res == EWSDecoder.RESULT_BLOCK_END:
                        print(" (End of block)", flush=True)
                    elif res is None:
                        pass
                    elif res == "":
                        print(" ", end="", flush=True)
                    else:
                        print(f" ({res}) ", end="", flush=True)
                except Exception as e:
                    print(f"\nError: {e}", flush=True)
                    self.decoder = EWSDecoder.decode()
                    self.decoder.__next__()
