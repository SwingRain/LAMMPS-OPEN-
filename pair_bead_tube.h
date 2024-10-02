
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(beadtube ,PairBeadTube)

#else

#ifndef LMP_PAIR_BEAD_TUBE_H
#define LMP_PAIR_BEAD_TUBE_H

#include "pair.h"
#include "my_page.h"
#include <math.h>
#include "math_const.h"

namespace LAMMPS_NS {

class PairBeadTube : public Pair {
 public:
  PairBeadTube(class LAMMPS *);
  virtual ~PairBeadTube();
  virtual void compute(int, int);
  void CNT_neigh();
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  void *extract(const char *, int &);
  double memory_usage();
  //NEED TO UPDATE PROTOTYPES:x


 protected:
  double cut_global;
  double **cut, **rad;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  int atomtypemax_cnt, bondtypemax_cnt, angletypemax_cnt;
  double TubeRad_cnt, cutoff_cnt;
  double epsilon_cnt, sigma_cnt, lj1_cnt, lj2_cnt, sixth_root_sig;
  double eps_by_2, pi_by_cut_sig;

  int maxlocal;                    // size of numneigh, firstneigh arrays
  int pgsize;                      // size of neighbor page
  int oneatom;                     // max # of neighbors for one atom
  MyPage<int> *ipage;              // neighbor list pages
  int *CNT_numneigh;              // # of pair neighbors for each atom
  int **CNT_firstneigh;           // ptr to 1st neighbor of each atom

  virtual void allocate();
};

}

#endif
#endif
