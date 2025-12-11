OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg meas[1];
x q[0];
u3(pi/2, -pi/4, -pi/4) q[0];
rz(pi/4) q[0];
measure q[0] -> meas[0];