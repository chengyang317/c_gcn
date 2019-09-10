# coding=utf-8
import visdom

__all__ = ['LineWin', 'TextWin', 'VisdomEnv']


visdom_pool = {}


class WinBase(object):
    def __init__(self, win_name, vis_env, win_opts: dict=None):
        self.win_name = win_name
        self.win_opts = {} if win_opts is None else win_opts
        self.vis_env = vis_env
        self.env_name = vis_env.env
        self.vis: visdom.Visdom = vis_env.vis
        self.window = None  # visdom window class instance


class LineWin(WinBase):
    def plot(self, x, y, train_tag, win_opts, is_reset):
        for key, value in win_opts.items():
            self.win_opts[key] = value
        if not self.window:
            self.win_opts['legend'] = [train_tag]
            self.window = self.vis.line(X=x, Y=y, env=self.env_name, opts=self.win_opts)
            self.win_opts.pop('legend')
        else:
            self.vis.line(y, x, self.window, update='append' if not is_reset else 'replace', name=train_tag)


class TextWin(WinBase):
    def plot(self, text, win=None, env=None, opts=None, append=False):
        win = win or self.window
        env = env or self.env_name
        opts = opts or self.win_opts
        if isinstance(text, dict):
            text = [f'{key}: {value}' for key, value in text.items()]
            text = '\n'.join(text)
        if win is None:
            self.window = self.vis.text(text, env=env, opts=opts, append=append)
        else:
            self.vis.text(text, win, env, opts, append)


class VisdomEnv(object):
    def __init__(self, server, port, env):
        self.server = server
        self.port = port
        self.env = env
        visdom_key = f'{str(server)}_{str(port)}_{env}'
        if visdom_key in visdom_pool:
            self.vis = visdom_pool[visdom_key]
        else:
            self.vis = visdom_pool[visdom_key] = visdom.Visdom(server=server, port=port, env=env, use_incoming_socket=False)
        self.wins = {}

    @classmethod
    def build(cls, args):
        server, port, env = args.logger_server, args.logger_port, args.logger_env or args.proj_name
        return cls(server, port, env)

    def new_line_win(self, win_name, win_opts=None):
        if win_name not in self.wins.keys():
            self.wins[win_name] = LineWin(win_name=win_name, vis_env=self, win_opts=win_opts)
        return self.wins[win_name]

    def new_text_win(self, win_name, win_opts=None):
        if win_name not in self.wins.keys():
            self.wins[win_name] = TextWin(win_name=win_name, vis_env=self, win_opts=win_opts)
        return self.wins[win_name]













