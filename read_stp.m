function data = read_stp(filename)
% READ_STH  Lee fichero .sth generado por escru
%   data = read_sth('caso_00001.sth')

fid = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

%% ── HEADER ──────────────────────────────────────────────────────────────
hid = H5G.open(fid, '/header');

data.nacum   = h5read(filename, '/header/nacum');
data.my      = h5read(filename, '/header/my');
data.mx      = h5read(filename, '/header/mx');
data.mz      = h5read(filename, '/header/mz');
data.time    = h5read(filename, '/header/time');
data.Re      = h5read(filename, '/header/Re');
data.alp     = h5read(filename, '/header/alp');
data.bet     = h5read(filename, '/header/bet');
data.vcon    = h5read(filename, '/header/vcon');
data.y       = h5read(filename, '/header/y');       % (my)
data.timev   = h5read(filename, '/header/timev');   % (nstp)
data.probe_j = h5read(filename, '/header/probe_j');  % (nprobes)


H5G.close(hid);

%% ── STATISTICS (my x nstp) ──────────────────────────────────────────────
data.utmp = h5read(filename, '/sta/utmp');   % (my, nstp)
data.vtmp = h5read(filename, '/sta/vtmp');
data.wtmp = h5read(filename, '/sta/wtmp');

%% ── PROBES (mgalz x nprobes x nstp) ────────────────────────────────────
data.probe_u = h5read(filename, '/sta/probe_u');  % (mgalz, nprobes, nstp)
data.probe_v = h5read(filename, '/sta/probe_v');
data.probe_w = h5read(filename, '/sta/probe_w');

H5F.close(fid);

fprintf('Leído: %s\n', filename);
fprintf('  my=%d  mx=%d  mz=%d  nacum=%d  nstp=%d\n', ...
        data.my, data.mx, data.mz, data.nacum, numel(data.timev));
end