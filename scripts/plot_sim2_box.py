import pandas as pd
import matplotlib.pyplot as plt

for r1 in [[128, 72, 48, 12]]:  # [0], [128, 72, 48, 12], [128, 84, 52, 14]
    input_path = "WGBS_ref_2+1.txt"  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"
    simi = 2
    mode = "high-resolution"
    for file in ["0.3False0.4"]:
        # 创建数据
        data = {
            'Reference': ['reference 1', 'reference 1', 'reference 2', 'reference 2'],
            'Simulation': ['simulation 1', 'simulation 2', 'simulation 1', 'simulation 2'],
            # 'CCC': [
            #     [0.7264336664568958, 0.7069703831636238, 0.8778625559927937, 0.661799652071601, 0.7465810165951237,
            #      0.6512128391561651, 0.4233958252359058, 0.6970052416566443, 0.7711926629286983],
            #     [0.6480075458132414, 0.6821716420168802, 0.820622612145469, 0.6028906365469792, 0.6833497737420993,
            #      0.6076434494503916, 0.2832246513103761, 0.6209315574333583, 0.7752596987518171],
            #     [0.49573524101585276, 0.44475941389141355, 0.6747643953790391, 0.43112706981377147, 0.7332343360697858,
            #      0.38989565999155074, 0.37588652663572075, 0.49438963574927014, 0.6247127001432442],
            #     [0.550739772260629, 0.5065550988597655, 0.7081470735695892, 0.46011820445991114, 0.7700146065109741,
            #      0.4163915043404567, 0.31056029013300757, 0.4552129696334257, 0.6213551408056553]
            # ]
            'CCC': [
                [0.7697207488334884, 0.6950814722385903, 0.8472835688763912, 0.6831452614005897, 0.8096056601037332,
                 0.7760572288655274, 0.5746163532135546, 0.6834125587626173, 0.8677926737417923],
                [0.696245121753054, 0.7115136246208968, 0.8106817455224351, 0.6035845198479745, 0.7510437460086773,
                 0.7586504715645751, 0.49079367954820013, 0.6491981513291135, 0.8869420341283493],
                [0.5427972695470527, 0.4041967425105771, 0.6537397679556666, 0.41683779974187535, 0.7447631208006836,
                 0.4666980654347037, 0.6995485603159425, 0.5564901456050334, 0.6481271209468191],
                [0.6159664949789707, 0.4910126171826903, 0.728765408446624, 0.41671893554153144, 0.8127714572165645,
                 0.4932994249322581, 0.7313534230935914, 0.5397167034740546, 0.6692623061305619]
            ]
        }

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 创建箱型图
        plt.figure(figsize=(10, 6))
        box_plot = plt.boxplot([df[df['Reference'] == ref][df['Simulation'] == sim]['CCC'].iloc[0]
                                for ref in ['reference 1', 'reference 2']
                                for sim in ['simulation 1', 'simulation 2']],
                               labels=['reference 1', 'reference 1',
                                       'reference 2', 'reference 2'],
                               patch_artist=True)  # 增加线条宽度

        # 自定义箱体颜色和边框
        colors = ['lightblue', 'lightpink', 'lightblue', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')  # 设置边框颜色
            patch.set_linewidth(2)  # 设置边框宽度

        # 加粗其他元素
        for whisker in box_plot['whiskers']:
            whisker.set_linewidth(2)
        for cap in box_plot['caps']:
            cap.set_linewidth(2)
        for median in box_plot['medians']:
            median.set_linewidth(2)

        # 自定义图表
        plt.title('WGBS-ours', fontsize=16, fontweight='bold')  # 加粗标题
        plt.ylabel('CCC', fontsize=12, fontweight='bold')  # 加粗y轴标签
        plt.ylim(0, 1)

        # 加粗坐标轴
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)

        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')

        # 添加图例
        plt.legend([plt.Rectangle((0, 0), 1, 1, fc='lightblue', ec='black', linewidth=2),
                    plt.Rectangle((0, 0), 1, 1, fc='lightpink', ec='black', linewidth=2)],
                   ['Simulation 1', 'Simulation 2'], loc='upper right')

        # 显示图表
        plt.tight_layout()
        plt.savefig('./results/figure/' + input_path + str(simi) + mode + file + str(r1) + '.png')
        plt.show()

