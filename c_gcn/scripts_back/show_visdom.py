# coding=utf-8
import pt_pack as pt
import argparse
from pathlib import Path



def visdom_log(logger_dir: Path):
    loggers = {
        'acc': pt.VisdomLogger('acc', str(logger_dir), env=logger_dir.name),
        'loss': pt.VisdomLogger('loss', str(logger_dir), env=logger_dir.name)
    }

    for logger in loggers.values():
        logger.show_log()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lllll')
    parser.add_argument('--proj_name', type=str)
    parser.add_argument('--proj_dir', type=str, default='./')
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    logger_dir = Path(f'{args.proj_dir}/work_dir/loggers')

    if args.prefix is not None:
        sub_dirs = logger_dir.glob(f'{args.prefix}*')
        for sub_dir in sub_dirs:
            visdom_log(sub_dir)





# loggers = {
#     'acc': pt.VisdomLogger('acc', f'{args.proj_dir}/work_dir/loggers/{args.proj_name}', env=args.proj_name),
#     'loss': pt.VisdomLogger('loss', f'{args.proj_dir}/work_dir/loggers/{args.proj_name}', env=args.proj_name)
# }
#
#
# for logger in loggers.values():
#     logger.show_log()







