from utils.plot_utils import *
from argparse import ArgumentParser
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2b", help="Base model name")
    parser.add_argument("--dataset", type=str, default="benchmark", choices=['benchmark', 'cat_harm'])
    args = parser.parse_args()

    plot_path = f'images/{args.dataset}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    if args.dataset == 'benchmark':
        bm_dir = f'cache/benchmarks'
        base_jb = pickle.load(open(os.path.join(bm_dir,f'{args.model}_base.pkl'), 'rb'))['jb']
        steer_jb = pickle.load(open(os.path.join(bm_dir,f'{args.model}_steer.pkl'), 'rb'))['jb']
        baseline_jb = pickle.load(open(os.path.join(bm_dir,f'{args.model}_baseline.pkl'), 'rb'))['jb']

        for k,v in base_jb.items():
            baseline_jb[k]['base'] = v
        for k,v in steer_jb.items():
            baseline_jb[k]['steer'] = v

        combined_safety_scores = defaultdict(dict)
        for k,v in baseline_jb.items():
            for kk,vv in v.items():
                combined_safety_scores[kk][k] = vv

    
        key_mapping = {
                        'cs': 'CosSim',
                        'act': 'ActDiff',
                        'la': 'AP',
                        'base': 'Base',
                        'steer': 'AS',
                        'our': 'CosSim+AP (Ours)',
                        }
        outer_keys = ['base','cs', 'act', 'la','our','steer']
        inner_keys = list(next(iter(combined_safety_scores.values())).keys())  # ['harmbench_test', 'jailbreakbench', 'advbench']

        num_groups = len(inner_keys)
        num_bars_per_group = len(outer_keys)

        # Set positions
        x = np.arange(num_groups)  # harmbench_test, jailbreakbench, advbench
        bar_width = 0.1  # adjust as needed

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot each outer key as one series of bars
        for i, outer_key in enumerate(outer_keys):
            values = [combined_safety_scores[outer_key][inner_key] for inner_key in inner_keys]
            ax.bar(x + i * bar_width, values, width=bar_width, label=key_mapping[outer_key])

        # Configure x-axis
        ax.set_xticks(x + bar_width * (num_bars_per_group - 1) / 2)
        ax.set_xticklabels([x.capitalize().split('_')[0].strip() for x in inner_keys],fontsize=14)
        # set y fontsize
        ax.tick_params(axis='y', labelsize=14)
        # ax.set_xlabel('Dataset')
        ax.set_ylabel('Safety',fontsize=14)
        # ax.set_title('Grouped Bar Plot')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend
        ax.legend(fontsize=12,ncol=3,loc = 'upper center')

        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Show plot
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
    else:
        cat_harm_scores = pickle.load(open(f'cache/{"gemma" if "gemma" in args.model else "llama"}_cat_harm_trf.pkl', 'rb'))

        sns.set(style="darkgrid")
        cat_group_mapping = {
            'Illegal Activity': 'Ill.Act',
            'Child Abuse': 'Child.A',
            'Hate/Harass/Violence': 'H/H/V',
            'Physical Harm': 'Phys.H',
            'Economic Harm': 'Econ.H',
            'Fraud/Deception': 'Fra/Dec',
            'Adult Content': 'Adult.C'
        }
        labels_mapping = {
            'specific': 'Specific',
            'common': 'Common',
            'transfer': 'Specific (Transfer)'
        }

        categories = list(cat_harm_scores.keys())
        subcats    = list(next(iter(cat_harm_scores.values())).keys())
        x          = np.arange(len(categories))
        bar_w      = 0.25
        offsets    = np.linspace(-bar_w, bar_w, len(subcats))
        fig, ax    = plt.subplots(1, 1, figsize=(7, 5))
        colors     = sns.color_palette("deep", len(subcats))  # seaborn colors

        # Draw bars for each subcat
        for j, sub in enumerate(subcats):
            heights = [cat_harm_scores[cat][sub] for cat in categories]
            ax.bar(x + offsets[j], heights, bar_w, label=labels_mapping[sub], color=colors[j])

        ax.set_xticks(x)
        ax.set_xticklabels([cat_group_mapping[x] for x in categories], rotation=25, ha='right', fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=13)

        # Y-axis label and legend
        ax.set_ylabel('Normalized Jailbreak', fontsize=14)
        ax.legend(loc='upper center', frameon=False, fontsize=12, ncol=3)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.15)

        plt.savefig(plot_path, dpi=300)


if __name__ == "__main__":
    main()
            