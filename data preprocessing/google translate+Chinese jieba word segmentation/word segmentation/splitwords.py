import re
s = "China's Legend Holdings will split its several business arms to go public on stock markets, the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，中国联想控股将分拆其多个业务部门在股市上市。"
uncn = re.compile(r'[\u0061-\u007a,\u0020]')

uncn = re.compile(r'[\u4e00-\u9fa5]')
en = "".join(uncn.findall(s.lower()))

print(en)
