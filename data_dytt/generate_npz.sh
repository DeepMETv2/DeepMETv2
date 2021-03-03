nohup python -u generate_npz.py -d dy -f 3 -n 6250 > dy3.log &
nohup python -u generate_npz.py -d dy -f 4 -n 6250 > dy4.log &
nohup python -u generate_npz.py -d dy -f 5 -n 6250 > dy5.log &
nohup python -u generate_npz.py -d dy -f 6 -n 6250 > dy6.log &

nohup python -u generate_npz.py -d tt -f 3 -n 6250 > tt3.log &
nohup python -u generate_npz.py -d tt -f 4 -n 6250 > tt4.log &
nohup python -u generate_npz.py -d tt -f 5 -n 6250 > tt5.log &
nohup python -u generate_npz.py -d tt -f 6 -n 6250 > tt6.log &
