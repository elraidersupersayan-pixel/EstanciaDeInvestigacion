%% Cargar
d = read_sth('P125_83pi_28074.stp');

%% Ver evolución temporal de u en punto de malla j=64
figure
plot(d.timev, d.utmp(64,:),'o')
xlabel('t'); ylabel('u'); title('u(j=64) vs tiempo')

%% Ver señal de sonda ip=3 en z=1
figure
plot(d.timev, squeeze(d.probe_u(1, 3, :)),'o')
xlabel('t'); ylabel('u_{probe}')

%% Ver perfil instantáneo en último paso
figure
plot(d.utmp(:, end), d.y)
xlabel('u'); ylabel('y')