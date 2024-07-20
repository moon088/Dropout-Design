# GPU Disabled
import os
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f'CUDA_VISIBLE_DEVICES is changed from {CUDA_VISIBLE_DEVICES} to {os.environ["CUDA_VISIBLE_DEVICES"]}')

try:
    import glob
    import itertools
    from jinja2 import Template
    import json
    import keras
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    import seaborn as sns

    import training

    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '/models')
    stat_figs_path = os.path.abspath(src_path + '/stat_figs')

    def save_weights():
        def collect(out):
            def load_model(h5, info):
                base, ext = os.path.splitext(os.path.basename(h5))

                dt = {}
                _, dt['net'], dt['nlayers'], dt['nnodes'], dt['ndrops'], dt['dropout'], dt['total_epochs'], dt['total_exps'], dt['dataset_name'], dt['nepochs'], dt['nexps'], _ = tuple(re.split(r'model_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)', base))
                dt['nlayers'] = int(dt['nlayers'])
                dt['nnodes'] = int(dt['nnodes'])
                dt['ndrops'] = int(dt['ndrops'])
                dt['total_epochs'] = int(dt['total_epochs'])
                dt['total_exps'] = int(dt['total_exps'])
                dt['nepochs'] = int(dt['nepochs'])
                dt['nexps'] = int(dt['nexps'])
                print(dt)

                model = keras.models.load_model(h5, custom_objects={'DropoutDesignLayer': training.DropoutDesignLayer})
                print(''.join(['loaded ', str(h5)]))
                dt['weights'] = [ w for w in model.get_weights()]

                info[base] = dt

            info = {}
            for h5 in glob.glob(models_path + '/model_*.h5'):
                load_model(h5, info)

            with open(out, 'w', newline='\n') as f:
                json.dump(info, f, indent=4, cls=NumpyArrayEncoder)
            return info

        # Summary
        def summarize(info, out):
            print('Collecting data')
            summary = {}
            for dt in info.values():

                key = '_'.join([str(dt['net']), str(dt['nlayers']), str(dt['nnodes']), str(dt['ndrops']), str(dt['dropout']), str(dt['total_epochs']), str(dt['total_exps']), str(dt['dataset_name']), str(dt['nepochs'])])
                if key not in summary:
                    sm = {
                        'net': dt['net'],
                        'nlayers': dt['nlayers'],
                        'nnodes': dt['nnodes'],
                        'ndrops': dt['ndrops'],
                        'dropout': dt['dropout'],
                        'total_epochs': dt['total_epochs'],
                        'total_exps': dt['total_exps'],
                        'dataset_name': dt['dataset_name'],
                        'nepochs': dt['nepochs'],
                    }
                    summary[key] = sm

                sm = summary[key]
                # nexps = dt['nexps']
                dweights = dt['weights']
                if 'weights' not in sm:
                    sweights = []
                    for i, wmat in enumerate(dweights):
                        sweights.append([])
                        for j, wv in enumerate(wmat):
                            if isinstance(wv, np.ndarray):
                                sweights[-1].append([[float(w)] for w in wv])
                            elif isinstance(wv, np.float32):
                                sweights[-1].append([float(wv)])
                            else:
                                raise TypeError(type(wv))
                    sm['weights'] = sweights
                else:
                    sweights = sm['weights']
                    for i, wmat in enumerate(dweights):
                        for j, wv in enumerate(wmat):
                            if isinstance(wv, np.ndarray):
                                for k, w in enumerate(wv):
                                    sweights[i][j][k].append(float(w))
                            elif isinstance(wv, np.float32):
                                sweights[i][j].append(float(wv))
                            else:
                                raise TypeError(type(wv))

            # statistics
            print('Collecting statistics')
            for sm in summary.values():
                weights_stat = []
                sweights = sm['weights']
                for i, wmat in enumerate(sweights):
                    weights_stat.append([])
                    for j, wv in enumerate(wmat):
                        assert isinstance(wv, list)
                        weights_stat[-1].append([])
                        if isinstance(wv[0], list):
                            for k, w in enumerate(wv):
                                weights_stat[-1][-1].append({
                                    'avg': float(np.average(w)),
                                    'var': np.var(w, dtype=float),
                                    'std': np.std(w, dtype=float)
                                })
                        elif isinstance(wv[0], float):
                            weights_stat[-1].append({
                                    'avg': float(np.average(wv)),
                                    'var': np.var(wv, dtype=float),
                                    'std': np.std(wv, dtype=float)
                                })
                        else:
                            raise TypeError(type(wv))
                sm['weights_stat'] = weights_stat
            with open(out, 'w', newline='\n') as f:
                json.dump(summary, f, indent=4, cls=NumpyArrayEncoder)

        def plot(out):
            with open(out, 'r') as f:
                summary = json.load(f)
            fig_paths = []
            for k, sm in summary.items():
                # plot standard devs.
                weights_std_lst = [list(itertools.chain.from_iterable((w['std'] for w in weight) for weight in weights)) for i, weights in enumerate(sm['weights_stat']) if i % 2 == 0 ]
                weights_std = np.zeros(shape=(len(weights_std_lst), max(len(v) for v in weights_std_lst)))
                for i, stds in enumerate(weights_std_lst):
                    for j, std in enumerate(stds):
                        weights_std[i][j] = std
                sns.heatmap(weights_std)
                if not os.path.exists(stat_figs_path):
                    os.mkdir(stat_figs_path)
                fig_path = stat_figs_path + '/' + k + '.png'
                plt.savefig(fig_path)
                fig_paths.append((k, fig_path))

            html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>重み推定値の標準偏差</title>
        </head>
        <body>
            <h1><font size=14>重み推定値の標準偏差</font></h1>
            <table border="1" >
                {% for row in rows %}
                    <tr>
                        <th>
                            <font size=12>{{row[0]}}<font>
                        </th>
                        <td>
                            <img border="0" src="{{row[1]}}"  alt="{{row[1]}}">
                        </td>
                    </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        '''
            template_data = {'rows' : fig_paths}
            template = Template(html)
            print(template.render(template_data))
            html_path = stat_figs_path + '/heatmaps.html'
            with open(html_path, 'w',  encoding='utf-8') as f:
                f.write(template.render(template_data))

        info_out = models_path + '/weights.json'
        stat_out = models_path + '/weights_stat.json'
        info = collect(info_out)
        summarize(info, stat_out)
        plot(stat_out)
    save_weights()
finally:
    # Restore GPU setting
    if CUDA_VISIBLE_DEVICES is None:
        if  "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    print(f'GPU setting recovered to {os.environ.get("CUDA_VISIBLE_DEVICES")}')