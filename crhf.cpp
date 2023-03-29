// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Libint Gaussian integrals library
#include <libint2.hpp>
#if !LIBINT2_CONSTEXPR_STATICS
#  include <libint2/statics_definition.h>
#endif
using real_t = libint2::scalar_type;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
		Matrix;

struct Atom {
	int atomic_number;
	double x, y, z;
};

//函数声明
std::vector<Atom> read_dotxyz(const std::string& filename);//在.xyz中读结构
/*
  传入.xyz文件，输出Atom类型
 */
std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms);
/*
  得到基函数信息，格式copy了libint的案例，得到一个shells
  电子积分的计算都是以壳层为单位去算，效率会高
 */
size_t nbasis(const std::vector<libint2::Shell>& shells);//基函数数目，用来size_t
/*
  积分计算引擎需要传入壳层数，最高角动量，基函数数目
 */
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
/*
  从壳层到基函数的映射
 */
Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,
					      libint2::Operator obtype,
						  const std::vector<Atom>& atoms = std::vector<Atom>());
/*
  只计算1e积分不需要onst std::vector<Atom>& atoms，这部分是用来计算核-电子势的
 */
//计算Fock矩阵
Matrix compute_K_Matrix(const std::vector<libint2::Shell>& shells,
	const Matrix& D);
Matrix compute_J_Matrix(const std::vector<libint2::Shell>& shells,
	const Matrix& D);

int main(int argc, char *argv[]){
	
	using std::cout;
	using std::cerr;
	using std::endl;
	
	using libint2::Shell;
	using libint2::Engine;
	using libint2::Operator;
	
	/*** =========================== ***/
	/*** initialize molecule         ***/
	/*** =========================== ***/
	
	// read geometry from a file; by default read from h2o.xyz, else take filename (.xyz) from the command line
	const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
	std::vector<Atom> atoms = read_dotxyz(filename);
	
	// count the number of electrons
	auto nelectron = 0;
	for (auto i = 0; i < atoms.size(); ++i)
		nelectron += atoms[i].atomic_number;
	const auto ndocc = nelectron / 2;
	//这里写死了，只能rhf，不能uhf
	
	// compute the nuclear repulsion energy
	//对照python部分重写的
	auto E_nuc = 0.0;
	for (auto i = 0; i < atoms.size(); i++)
		for (auto j = i + 1; j < atoms.size(); j++) {
		auto xij = atoms[i].x - atoms[j].x;
		auto yij = atoms[i].y - atoms[j].y;
		auto zij = atoms[i].z - atoms[j].z;
		auto r2 = xij*xij + yij*yij + zij*zij;
		auto r = sqrt(r2);
		E_nuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
	}
	    cout << "\tNuclear repulsion energy = " << E_nuc << endl;
	
	/*** =========================== ***/
	/*** create basis set            ***/
	/*** =========================== ***/
	auto shells = make_sto3g_basis(atoms);
	size_t nao = 0;
	for (auto s=0; s<shells.size(); ++s)
		nao += shells[s].size();
	//nao和rhf.py中的含义相同
	
	/*** =========================== ***/
	/*** compute 1-e integrals       ***/
	/*** =========================== ***/
	
	// initializes the Libint integrals library ... now ready to compute
	libint2::initialize();
	
	// compute overlap integrals
	auto S = compute_1body_ints(shells, Operator::overlap);
	//cout << "\n\tOverlap Integrals:\n";
	//cout << S << endl;
	
	// compute kinetic-energy integrals
	auto T = compute_1body_ints(shells, Operator::kinetic);
	//cout << "\n\tKinetic-Energy Integrals:\n";
	//cout << T << endl;
	
	// compute nuclear-attraction integrals
	Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);
	//cout << "\n\tNuclear Attraction Integrals:\n";
	//cout << V << endl;
	
	// Core Hamiltonian = T + V
	Matrix H = T + V;
	//cout << "\n\tCore Hamiltonian:\n";
	//cout << H << endl;
	
	// T and V no longer needed, free up the memory
	T.resize(0,0);
	V.resize(0,0);

	/*** =========================== ***/
	/*** build initial-guess density ***/
	/*** =========================== ***/
	
	Matrix D;
	//简单使用核初猜
	// solve H C = e S C
	Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(H, S);
	auto eps = gen_eig_solver.eigenvalues();
	auto C = gen_eig_solver.eigenvectors();
	//cout << "\n\tInitial C Matrix:\n";
	//cout << C << endl;
	
	// compute density, D = C(occ) . C(occ)T
	auto C_occ = C.leftCols(ndocc);
	D = C_occ * C_occ.transpose();
	
	//cout << "\n\tInitial Density Matrix:\n";
    //cout << D << endl;

	
	/*** =========================== ***/
	/*** main iterative loop         ***/
	/*** =========================== ***/
    //开始scf循环了，和rhf.py流程一样的，区别是没有diis
	
	//收敛限设置，和rhf.py一样
	const auto max_iter = 100;
	const real_t E_conv = 1e-10;
	const real_t D_conv = 1e-8;
	auto iter = 0;
	real_t dRMS = 0.0;
	real_t dE = 0.0;
	real_t SCF_E = 0.0;
	
	do {
		const auto tstart = std::chrono::high_resolution_clock::now();
		++iter;
		
		// Save a copy of the energy and the density
		auto E_last = SCF_E;
		auto D_last = D;
		
		// build a new Fock matrix
		auto F = H;
		F += compute_J_Matrix(shells, D);
		F += compute_K_Matrix(shells, D);
		
		// solve F C = e S C
		Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
		auto eps = gen_eig_solver.eigenvalues();
		auto C = gen_eig_solver.eigenvectors();
		
		// compute density, D = C(occ) . C(occ)T
		auto C_occ = C.leftCols(ndocc);
		D = C_occ * C_occ.transpose();
		
		// compute HF energy
		SCF_E = 0.0;
		for (auto i = 0; i < nao; i++)
			for (auto j = 0; j < nao; j++)
				SCF_E += D(i,j) * (H(i,j) + F(i,j));
		
		// compute difference with last iteration
		dE = SCF_E - E_last;
		dRMS = (D - D_last).norm();
		
		const auto tstop = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double> time_elapsed = tstop - tstart;

		if (iter == 1)
			std::cout <<
			"\n\n Iter        E(elec)              E(tot)               Delta(E)             RMS(D)         Time(s)\n";
		printf(" %02d %20.12f %20.12f %20.12f %20.12f %10.5lf\n", iter, SCF_E, SCF_E + E_nuc,
			dE, dRMS, time_elapsed.count());
		
	} while (((fabs(dE) > E_conv) || (fabs(dRMS) > D_conv)) && (iter < max_iter));
	
	printf("** Hartree-Fock energy = %20.12f\n", SCF_E + E_nuc);
	
	libint2::finalize(); // done with libint
	return 0;
}

//函数实现
std::vector<Atom> read_dotxyz(const std::string& filename) {
	
	std::cout << "Will read geometry from " << filename << std::endl;
	std::ifstream is(filename);
	assert(is.good());
	
	// line 1 = # of atoms
	size_t natom;
	is >> natom;
	// read off the rest of line 1 and discard
	std::string rest_of_line;
	std::getline(is, rest_of_line);
	
	// line 2 = comment (possibly empty)
	std::string comment;
	std::getline(is, comment);
	
	std::vector<Atom> atoms(natom);
	for (auto i = 0; i < natom; i++) {
		std::string element_label;
		double x, y, z;
		is >> element_label >> x >> y >> z;
		
		// .xyz files report element labels, hence convert to atomic numbers
		int Z;
		if (element_label == "H")
			Z = 1;
		else if (element_label == "C")
			Z = 6;
		else if (element_label == "N")
			Z = 7;
		else if (element_label == "O")
			Z = 8;
		else if (element_label == "F")
			Z = 9;
		else if (element_label == "S")
			Z = 16;
		else if (element_label == "Cl")
			Z = 17;
		else {
			std::cerr << "read_dotxyz: element label \"" << element_label << "\" is not recognized" << std::endl;
			throw std::invalid_argument("Did not recognize element label in .xyz file");
		}
		
		atoms[i].atomic_number = Z;
		
	    //单位换算
		const auto angstrom_to_bohr = 1 / 0.52917721092;
		atoms[i].x = x * angstrom_to_bohr;
		atoms[i].y = y * angstrom_to_bohr;
		atoms[i].z = z * angstrom_to_bohr;
	}
	
	return atoms;
}

std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms){
	
	using libint2::Shell;
	
	std::vector<Shell> shells;
	
	for(auto a=0; a<atoms.size(); ++a) {
		
		// STO-3G basis set
		// cite: W. J. Hehre, R. F. Stewart, and J. A. Pople, The Journal of Chemical Physics 51, 2657 (1969)
		//       doi: 10.1063/1.1672392
		// obtained from https://bse.pnl.gov/bse/portal
		switch (atoms[a].atomic_number) {
			case 1: // Z=1: hydrogen
			shells.push_back(
				{
					{3.425250910, 0.623913730, 0.168855400}, // exponents of primitive Gaussians
					{  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
						{0, false, {0.15432897, 0.53532814, 0.44463454}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
				}
				);
			break;
			
			case 6: // Z=6: carbon
			shells.push_back(
				{
					{71.616837000, 13.045096000, 3.530512200},
					{
						{0, false, {0.15432897, 0.53532814, 0.44463454}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{2.941249400, 0.683483100, 0.222289900},
					{
						{0, false, {-0.09996723, 0.39951283, 0.70011547}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{2.941249400, 0.683483100, 0.222289900},
					{ // contraction 0: p shell (l=1), spherical=false
						{1, false, {0.15591627, 0.60768372, 0.39195739}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			break;
			
			case 7: // Z=7: nitrogen
			shells.push_back(
				{
					{99.106169000, 18.052312000, 4.885660200},
					{
						{0, false, {0.15432897, 0.53532814, 0.44463454}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{3.780455900, 0.878496600, 0.285714400},
					{
						{0, false, {-0.09996723, 0.39951283, 0.70011547}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{3.780455900, 0.878496600, 0.285714400},
					{ // contraction 0: p shell (l=1), spherical=false
						{1, false, {0.15591627, 0.60768372, 0.39195739}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			break;
			
			case 8: // Z=8: oxygen
			shells.push_back(
				{
					{130.709320000, 23.808861000, 6.443608300},
					{
						{0, false, {0.15432897, 0.53532814, 0.44463454}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{5.033151300, 1.169596100, 0.380389000},
					{
						{0, false, {-0.09996723, 0.39951283, 0.70011547}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			shells.push_back(
				{
					{5.033151300, 1.169596100, 0.380389000},
					{ // contraction 0: p shell (l=1), spherical=false
						{1, false, {0.15591627, 0.60768372, 0.39195739}}
					},
					{{atoms[a].x, atoms[a].y, atoms[a].z}}
				}
				);
			break;
			
		default:
			throw "do not know STO-3G basis for this Z";
		}
		
	}
	
	return shells;
}

size_t nbasis(const std::vector<libint2::Shell>& shells) {
	size_t n = 0;
	for (const auto& shell: shells)
		n += shell.size();
	return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
	size_t n = 0;
	for (auto shell: shells)
		n = std::max(shell.nprim(), n);
	return n;
}

int max_l(const std::vector<libint2::Shell>& shells) {
	int l = 0;
	for (auto shell: shells)
		for (auto c: shell.contr)
			l = std::max(c.l, l);
	return l;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
	std::vector<size_t> result;
	result.reserve(shells.size());
	
	size_t n = 0;
	for (auto shell: shells) {
		result.push_back(n);
		n += shell.size();
	}
	
	return result;
}

Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,
						  libint2::Operator obtype,
					      const std::vector<Atom>& atoms)
{
	using libint2::Shell;
	using libint2::Engine;
	using libint2::Operator;
	
	const auto n = nbasis(shells);
	Matrix result(n,n);
	
	// construct the overlap integrals engine
	Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
	
	//下面都是做核吸引积分的计算
	if (obtype == Operator::nuclear) {
		std::vector<std::pair<real_t,std::array<real_t,3>>> q;
		/*
		  声明了一个名为 "q" 的向量容器（vector container），其中每个元素都是一个pair，包含两个成员：
		  一个名为 "real_t" 的实数类型，以及一个由三个实数类型组成的数组（array）
		 */
		for(const auto& atom : atoms) {
			//遍历atoms，把原子序数 和 xyz存到q里面
			q.push_back( {static_cast<real_t>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
		}
		engine.set_params(q);//把q再传给engine
	}
	
	auto shell2bf = map_shell_to_basis_function(shells);
	
	// buf[0] points to the target shell set after every call  to engine.compute()
	const auto& buf = engine.results();
	
	for(auto s1=0; s1!=shells.size(); ++s1) {
		auto bf1 = shell2bf[s1];
		auto n1 = shells[s1].size();
		
		for(auto s2=0; s2<=s1;++s2) {
			auto bf2 = shell2bf[s2];
			auto n2 = shells[s2].size();
			
			// compute shell pair
			engine.compute(shells[s1], shells[s2]);
			
			// "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
			Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
			result.block(bf1, bf2, n1, n2) = buf_mat;//Eigen库的函数
			if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
				result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
			//应用了矩阵对称性
		}
	}
	
	return result;
}

Matrix compute_J_Matrix(const std::vector<libint2::Shell>& shells, 
						const Matrix& D)
{
	
	using libint2::Shell;
	using libint2::Engine;
	using libint2::Operator;
	
	const auto n = nbasis(shells);
	Matrix J = Matrix::Zero(n,n);
	
	// construct the electron repulsion integrals engine
	Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
	
	auto shell2bf = map_shell_to_basis_function(shells);
	
	// buf[0] points to the target shell set after every call  to engine.compute()
	const auto& buf = engine.results();
	
	// loop over shell pairs of the Fock matrix, {s1,s2}
	// Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
	for(auto s1=0; s1!=shells.size(); ++s1) {
		
		auto bf1_first = shell2bf[s1]; // first basis function in this shell
		auto n1 = shells[s1].size();
		
		for(auto s2=0; s2!=shells.size(); ++s2) {
			
			auto bf2_first = shell2bf[s2];
			auto n2 = shells[s2].size();
			
			// loop over shell pairs of the density matrix, {s3,s4}
			// again symmetry is not used for simplicity
			for(auto s3=0; s3!=shells.size(); ++s3) {
				
				auto bf3_first = shell2bf[s3];
				auto n3 = shells[s3].size();
				
				for(auto s4=0; s4!=shells.size(); ++s4) {
					
					auto bf4_first = shell2bf[s4];
					auto n4 = shells[s4].size();
					
					// Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
					engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
					const auto* buf_1234 = buf[0];
					if (buf_1234 == nullptr)
						continue; // if all integrals screened out, skip to next quartet
					
					for(auto f1=0, f1234=0; f1!=n1; ++f1) {
						const auto bf1 = f1 + bf1_first;
						for(auto f2=0; f2!=n2; ++f2) {
							const auto bf2 = f2 + bf2_first;
							for(auto f3=0; f3!=n3; ++f3) {
								const auto bf3 = f3 + bf3_first;
								for(auto f4=0; f4!=n4; ++f4, ++f1234) {
									const auto bf4 = f4 + bf4_first;
									J(bf1,bf2) += D(bf3,bf4) * 2.0 *buf_1234[f1234];
								}
							}
						}
					}
				
				}
			}
		}	
	}
	return J;
}

Matrix compute_K_Matrix(const std::vector<libint2::Shell>& shells,
						const Matrix& D)
{
	
	using libint2::Shell;
	using libint2::Engine;
	using libint2::Operator;
	
	const auto n = nbasis(shells);
	Matrix K = Matrix::Zero(n,n);
	
	// construct the electron repulsion integrals engine
	Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
	
	auto shell2bf = map_shell_to_basis_function(shells);
	
	// buf[0] points to the target shell set after every call  to engine.compute()
	const auto& buf = engine.results();
	
	// loop over shell pairs of the Fock matrix, {s1,s2}
	// Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
	for(auto s1=0; s1!=shells.size(); ++s1) {
		
		auto bf1_first = shell2bf[s1]; // first basis function in this shell
		auto n1 = shells[s1].size();
		
		for(auto s2=0; s2!=shells.size(); ++s2) {
			
			auto bf2_first = shell2bf[s2];
			auto n2 = shells[s2].size();
			
			// loop over shell pairs of the density matrix, {s3,s4}
			// again symmetry is not used for simplicity
			for(auto s3=0; s3!=shells.size(); ++s3) {
				
				auto bf3_first = shell2bf[s3];
				auto n3 = shells[s3].size();
				
				for(auto s4=0; s4!=shells.size(); ++s4) {
					
					auto bf4_first = shell2bf[s4];
					auto n4 = shells[s4].size();
					
					// exchange contribution to the Fock matrix is from {s1,s3,s2,s4} integrals
					engine.compute(shells[s1], shells[s3], shells[s2], shells[s4]);
					const auto* buf_1324 = buf[0];
					
					for(auto f1=0, f1324=0; f1!=n1; ++f1) {
						const auto bf1 = f1 + bf1_first;
						for(auto f3=0; f3!=n3; ++f3) {
							const auto bf3 = f3 + bf3_first;
							for(auto f2=0; f2!=n2; ++f2) {
								const auto bf2 = f2 + bf2_first;
								for(auto f4=0; f4!=n4; ++f4, ++f1324) {
									const auto bf4 = f4 + bf4_first;
									K(bf1,bf2) -= D(bf3,bf4) * buf_1324[f1324];
								}
							}
						}
					}
					
				}
			}
		}
	}
	
	return K;
}
/*
  第一个循环变量 f1，它的范围是 [0, n1)，每次循环迭代时，变量 f1 会自增 1。该循环用于遍历数组 buf_1234 中第一个维度的元素。
  变量 bf1 是一个基于 f1 计算得到的辅助变量，它等于 f1 + bf1_first。这里的 bf1_first 是一个整数常量，用于指定数组 I 中第一维的起始位置。
  
  第二个循环变量 f2，它的范围是 [0, n2)，每次循环迭代时，变量 f2 会自增 1。该循环用于遍历数组 buf_1234 中第二个维度的元素。
  变量 bf2 是一个基于 f2 计算得到的辅助变量，它等于 f2 + bf2_first。这里的 bf2_first 是一个整数常量，用于指定数组 I 中第二维的起始位置。
  
  第三个循环变量 f3，它的范围是 [0, n3)，每次循环迭代时，变量 f3 会自增 1。该循环用于遍历数组 buf_1234 中第三个维度的元素。
  变量 bf3 是一个基于 f3 计算得到的辅助变量，它等于 f3 + bf3_first。这里的 bf3_first 是一个整数常量，用于指定数组 I 中第三维的起始位置。
  
  第四个循环变量 f4，它的范围是 [0, n4)，每次循环迭代时，变量 f4 会自增 1。该循环用于遍历数组 buf_1234 中第四个维度的元素。
  变量 f1234 是一个基于 f1, f2, f3, f4 四个变量计算得到的辅助变量，它等于 f1 * n2 * n3 * n4 + f2 * n3 * n4 + f3 * n4 + f4。该变量用于将一维数组 buf_1234 中的元素映射到二维数组 I 中的特定位置。
  变量 bf4 是一个基于 f4 计算得
 */

