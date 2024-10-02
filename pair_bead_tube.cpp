/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Shengfeng Cheng and Junwen Wang (Virginia Tech)
   bead-tube potential for CNT-CNT and CNT-polymer interactions.
   A CNT is represented as a jointed-tube chain.
   A polymer chain is represented as a bead-spring chain.
   This bead-tube potential is for the interaction between a CNT segment
   (i.e., a tube) and another CNT bead or a polymer bead.
   Therefore, it is basically a 3-body interaction.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pair_bead_tube.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define PGDELTA 1

/* ---------------------------------------------------------------------- */

PairBeadTube::PairBeadTube(LAMMPS *lmp) : Pair(lmp)
{

  // respa_enable = 1;
  writedata = 1;

  maxlocal = 0;
  CNT_numneigh = NULL;
  CNT_firstneigh = NULL;
  ipage = NULL;
  pgsize = oneatom = 0;

}

/* ---------------------------------------------------------------------- */

PairBeadTube::~PairBeadTube()
{
  if (allocated) {

    memory->destroy(CNT_numneigh);
    memory->sfree(CNT_firstneigh);
    delete [] ipage;

    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(rad);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairBeadTube::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,m;

  double evdwl,fpair,fpair1,fpair2,fpair3;
  double rsq,r2inv,r6inv,forcelj,factor_lj;

  //TubeRad_cnt needs to be defined in header
  double rscaled, rscaled1, rscaled3;

  int *ilist,*CNT_neighs;
  int *force_comput_flag;

  // construct neighbor list of owned+ghost atoms
  // code for CNT_neigh() copied from PairAIREBO.cpp
  // NOTE: global cutoff is used for neighborlist construction
  // so it has to be big enough so no interactions are missing
  // global_cutoff > factor*sqrt((bond_len/2)**2+(TubeRad_cnt + cutoff_cnt)**2)
  // factor should be at least 1.2 to be safe
  CNT_neigh();

  // evdwl = 0.0;
  // if (eflag || vflag) ev_setup(eflag,vflag);
  // else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  tagint *tag = atom->tag;
  // bigint natoms = atom->natoms;
  int newton_bond = force->newton_bond;

  // flag of local + ghost atoms to make sure they are not double-counted in bead-tube interaction
  // we use a global cutoff big enough for the construction of neighborlist so that neighbors
  // of ghost atoms are all included in nall = nlocal + nghost atoms.
  // the point is that neighbors of ghost atoms identified in CNT_neigh are all from
  // nall atoms associated to a local processor as determined in neighbor list construction
  // if cutoff used in neighbor list construction is too small, then ghost atoms will miss some
  // neighbors, which is not what we want here
  force_comput_flag = new int [nall];

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  // int *num_bond = atom->num_bond;
  // tagint **bond_atom = atom->bond_atom;
  int i1, i2, i3, bondtype;
  tagint i1tag, i2tag, i3tag, jtag;
  tagint *molecule = atom->molecule;

  int force_ignore_flag;

  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int *num_angle = atom->num_angle;
  // tagint **angle_atom;
  int angletype;

  double x1tmp, y1tmp, z1tmp, x2tmp, y2tmp, z2tmp;
  double x3tmp, y3tmp, z3tmp, x0tmp, y0tmp, z0tmp;
  double x4tmp, y4tmp, z4tmp;
  double xcnt, ycnt, zcnt, rcnt, rcnt_sq;
  double xcnt1, ycnt1, zcnt1, rcnt1, rcnt1_sq;
  double xcnt3, ycnt3, zcnt3, rcnt3, rcnt3_sq;
  double del1x, del1y, del1z, del1, del1_sq;
  double del2x, del2y, del2z, del2, del2_sq;
  double del3x, del3y, del3z, del3, del3_sq;
  double del4x, del4y, del4z, del4, del4_sq;
  double hbeadtube, fscale = 1.0;
  double theta1, theta2, theta3;
  double del1_ratio, del2_ratio, del3_ratio;

  double sixth_root_sig = pow(2.0, 0.16666666) * sigma_cnt;
  double eps_by_2 = 0.5 * epsilon_cnt;
  double lj1_cnt = 48.0 * epsilon_cnt * pow(sigma_cnt, 12.0);
  double lj2_cnt = 24.0 * epsilon_cnt * pow(sigma_cnt, 6.0);
  double pi_by_cut_sig = MY_PI/(cutoff_cnt - sixth_root_sig);
  double arg, c;

  if (cutoff_cnt <= sixth_root_sig) {
     error->one(FLERR,"cutoff for BeadTube potential is too small");
  }

  // bond type "1 ... bondtypemax_cnt" must be all for CNT bonds
  double blen = 0;
  for (m = 1; m<= bondtypemax_cnt; m++) {
       c = force->bond->equilibrium_distance(m);
       if (c > blen) blen = c;
  }
  if (TubeRad_cnt + cutoff_cnt >= 0.5*blen) {
     error->warning(FLERR, "CNT radius and BeadTube potential cutoff are too large for CNT bond length");
  }
  // below comparison to make sure that global cutoff, which is used in the construction of neigbor lists,
  // is big enough so that all needed neighbors of a ghost atom (the i2 bead of a bond i1--i2 stored on the
  // processor where i1 is local but i2 is not) in the calculation of the BeadTube potential will be included
  // in the neighbor lists.
  double desired_cut_global = 1.3*( sqrt(pow(0.5*blen, 2.0) + pow(TubeRad_cnt + cutoff_cnt, 2.0)) + blen );
  if (cut_global <= desired_cut_global) {
     error->one(FLERR,"global cutoff used in constructing neighbor lists for BeadTube potential is too small");
  }

  // loop over all CNT bonds for bead-tube interactions
  for (ii = 0; ii < nbondlist; ii++) {

     bondtype = bondlist[ii][2];
     if(bondtype > bondtypemax_cnt || bondtype <= 0) continue; // CNT bond type must be from 1 to bondtypemax_cnt

     //Gets two beads belonging to same bond

     i1 = bondlist[ii][0];
     // i1 &= NEIGHMASK;
     i1tag = tag[i1];
     x1tmp = x[i1][0];
     y1tmp = x[i1][1];
     z1tmp = x[i1][2];

     i2 = bondlist[ii][1];
     // i2 &= NEIGHMASK;
     i2tag = tag[i2];
     x2tmp = x[i2][0];
     y2tmp = x[i2][1];
     z2tmp = x[i2][2];

     // note that i1 is always a local atom but i2 may be not.

     // bondlist is the information of all bonds owned by the local processor.

     // If newton_bond is ON, then a bond is stored once on a processor
     // in this case, i1 is always smaller than nlocal while i2 can be smaller or larger than nlocal
     // depending on if i2 is a local atom or a ghost aom on that processor.

     // if newton_bond is OFF, then a bond is stored only on one processor if both i1 and i2 are local.
     // if on a processor there is a bond with i1 < nlocal but i2 > nlocal, then the same bond is also stored
     // on the processor on which i2 is local. On that processor, i2 (stored as i1) < nlocal and i1 (stored as i2) > nlocal.

     // all these guarantee that the bonded interaction is correctly calculated no matter
     // newton_bond is ON or OFF.

     // the following command is unnecessary as i2 is guaranteed to be the closest image of i1
     // i2 = domain->closest_image(i1, i2);

     // CNT bond length, which is the central axis of the tube under consideration
     // vector pointing from bead 1 to bead 2
     xcnt = x2tmp - x1tmp;
     ycnt = y2tmp - y1tmp;
     zcnt = z2tmp - z1tmp;
     rcnt_sq = xcnt*xcnt + ycnt*ycnt + zcnt*zcnt;
     rcnt = sqrt(rcnt_sq);

     // use the following flag to make sure that neighbors shared by bead i1 and i2
     // are not counted twice when computing bead-tube interactions
     for (k = 0; k< nall; k++) {
          force_comput_flag[k] = 0;
     }

     // loop over all neighbors of bead i1 and i2
     // if newton_bond is ON, then each bond is only stored once and thus there is no double-counting
     // if newton_bond is OFF but both i1 and i2 are less than nlocal, then the bond is local and there is no double-counting
     // if newton_bond is OFF but i2 is not local (note i1 is always local but i2 may be not), then the bond is also stored
     // on the processor where i2 is local. In this case there is a double-counting of the bead-tube interactions involving
     // this bond. So a discounting factor 0.5 is introduced below to compensate for this double-counting.
     // the following code is not needed as now we require newton_bond ON in BeadTube pair style.
     // if (newton_bond) {
     //   fscale = 1.0; // newton_bond is ON, a bond is only stored once
     // }
     // else {
     //   if (i1 < nlocal && i2 < nlocal) {
     //      fscale = 1.0;  // local bonds, only stored on local processor
     //   }
     //   else {
     //       fscale = 0.5; // bonds stored twice on two different processors
     //                           // bead-tube interactions will be computed on both processors and thus be double-counted
     //                           // no communication is needed as interaction is reduced to 50% on each processor
     //   }
     // }

     // fprintf (screen, "fscale = %f\n", fscale);

    // loop over all neighbors of i1
    CNT_neighs = CNT_firstneigh[i1];
    for (k = 0; k < CNT_numneigh[i1]; k++) {

         j = CNT_neighs[k];
         jtag = tag[j];

         // fprintf(screen, "%i %i %i\n", i1tag, i2tag, jtag);

         force_comput_flag[j] = 1; // make sure this bead is not counted again if it is also a neighbor of i2

         force_ignore_flag = 0;
         // check if target bead and bead i1, i2 are connected by bonds
         // if yes, ignore target bead in bead-tube interaction
         if (molecule[i1] == molecule[j]) {
            if (abs(jtag - i1tag) == 1 || abs(jtag - i2tag) == 1) {
               force_ignore_flag = 1;
               // fprintf (screen, "Warning: Found bonded atoms in neighbor list 1: %i %i %i \n", i1tag, i2tag, jtag);
            }
         }
         if (force_ignore_flag) continue; // ignore beads directly connected to i1 or i2

         x0tmp = x[j][0];
         y0tmp = x[j][1];
         z0tmp = x[j][2];

         // vector pointing from bead 1 to target bead 0
         del1x = x0tmp - x1tmp;
         del1y = y0tmp - y1tmp;
         del1z = z0tmp - z1tmp;
         del1_sq = del1x*del1x + del1y*del1y + del1z*del1z;
         del1 = sqrt(del1_sq);

         // vector pointing from bead 2 to target bead 0
         del2x = x0tmp - x2tmp;
         del2y = y0tmp - y2tmp;
         del2z = z0tmp - z2tmp;
         del2_sq = del2x*del2x + del2y*del2y + del2z*del2z;
         del2 = sqrt(del2_sq);

         c = (del1_sq + rcnt_sq - del2_sq)/(2*del1*rcnt);
         if (c > 1.0) c = 1.0;
         if (c < -1.0) c = -1.0;
         theta1 = acos(c);

         c = (del2_sq + rcnt_sq - del1_sq)/(2*del2*rcnt);
         if (c > 1.0) c = 1.0;
         if (c < -1.0) c = -1.0;
         theta2 = acos(c);

         if (theta1 >= MY_PI2) { // pair-wise interaction between target bead and bead i1
            hbeadtube = del1;
            rscaled = hbeadtube - TubeRad_cnt;
            if (rscaled < cutoff_cnt) {
               if (rscaled < sixth_root_sig) { // repulsive LJ core
                  r2inv = 1.0/(rscaled * rscaled);
                  r6inv = r2inv*r2inv*r2inv;
                  forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                  fpair = fscale*forcelj/rscaled;
               }
               else { // attractive soft tail
                  arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                  fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
               }
               fpair1 = fpair/del1;
               f[i1][0] -= fpair1*del1x;
               f[i1][1] -= fpair1*del1y;
               f[i1][2] -= fpair1*del1z;

               f[j][0] += fpair1*del1x;
               f[j][1] += fpair1*del1y;
               f[j][2] += fpair1*del1z;
            }
         }
        else if (theta2 >= MY_PI2) { // pair-wise interaction between target bead and bead i2
            hbeadtube = del2;
            rscaled = hbeadtube - TubeRad_cnt;
            if (rscaled < cutoff_cnt) {
               if (rscaled < sixth_root_sig) { // repulsive LJ core
                  r2inv = 1.0/(rscaled * rscaled);
                  r6inv = r2inv*r2inv*r2inv;
                  forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                  fpair = fscale*forcelj/rscaled;
               }
               else { // attractive soft tail
                  arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                  fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
               }
               fpair2 = fpair/del2;
               f[i2][0] -= fpair2*del2x;
               f[i2][1] -= fpair2*del2y;
               f[i2][2] -= fpair2*del2z;

               f[j][0] += fpair2*del2x;
               f[j][1] += fpair2*del2y;
               f[j][2] += fpair2*del2z;
            }
         }
         else { // 3-body interaction amongst target bead, bead i1, and bead i2
            hbeadtube = del1 * sin(theta1); // can also be written as del2 * sin(theta2)
            rscaled = hbeadtube - TubeRad_cnt;
            if (rscaled < cutoff_cnt) {
               if (rscaled < sixth_root_sig) { // repulsive LJ core
                  r2inv = 1.0/(rscaled * rscaled);
                  r6inv = r2inv*r2inv*r2inv;
                  forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                  fpair = fscale*forcelj/rscaled;
               }
               else { // attractive soft tail
                  arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                  fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
               }

               // decide the base of the height of the triangle formed by i1, i2, and target bead
               del1_ratio = del1*cos(theta1)/rcnt;
               del2_ratio = 1 - del1_ratio;
               x3tmp = x1tmp + del1_ratio * xcnt;
               y3tmp = y1tmp + del1_ratio * ycnt;
               z3tmp = z1tmp + del1_ratio * zcnt;

               // vector pointing from base to target bead
               del3x = x0tmp - x3tmp;
               del3y = y0tmp - y3tmp;
               del3z = z0tmp - z3tmp;
               del3_sq = del3x*del3x + del3y*del3y + del3z*del3z;
               del3 = sqrt(del3_sq);

               // use lever rule to distribute force to i1 and i2 beads
               fpair1 = fpair/del3*del2_ratio;
               f[i1][0] -= fpair1*del3x;
               f[i1][1] -= fpair1*del3y;
               f[i1][2] -= fpair1*del3z;

               fpair2 = fpair/del3*del1_ratio;
               f[i2][0] -= fpair2*del3x;
               f[i2][1] -= fpair2*del3y;
               f[i2][2] -= fpair2*del3z;

               f[j][0] += fpair*del3x/del3;
               f[j][1] += fpair*del3y/del3;
               f[j][2] += fpair*del3z/del3;
            }
         }

    } // end of loop over neighbors of i1

    // loop over all neighbors of i2
    CNT_neighs = CNT_firstneigh[i2];
    for (k = 0; k < CNT_numneigh[i2]; k++) {

         j = CNT_neighs[k];
         jtag = tag[j];

         force_ignore_flag = 0;
         // check if target bead and bead i1, i2 are connected by bonds
         // if yes, ignore target bead in bead-tube interaction
         if (molecule[i1] == molecule[j]) {
            if (abs(jtag - i1tag) == 1 || abs(jtag - i2tag) == 1) {
               force_ignore_flag = 1;
               // fprintf (screen, "Warning: Found bonded atoms in neighbor list 2: %i %i %i \n", i1tag, i2tag, jtag);
            }
         }
         if (force_ignore_flag) continue; // ignore beads directly connected to i1 or i2

         if (force_comput_flag[j] == 0) { // make sure this bead is not counted again if it is already a neighbor of i1

            x0tmp = x[j][0];
            y0tmp = x[j][1];
            z0tmp = x[j][2];

            // vector pointing from bead 1 to target bead 0
            del1x = x0tmp - x1tmp;
            del1y = y0tmp - y1tmp;
            del1z = z0tmp - z1tmp;
            del1_sq = del1x*del1x + del1y*del1y + del1z*del1z;
            del1 = sqrt(del1_sq);

            // vector pointing from bead 2 to target bead 0
            del2x = x0tmp - x2tmp;
            del2y = y0tmp - y2tmp;
            del2z = z0tmp - z2tmp;
            del2_sq = del2x*del2x + del2y*del2y + del2z*del2z;
            del2 = sqrt(del2_sq);

            c = (del1_sq + rcnt_sq - del2_sq)/(2*del1*rcnt);
            if (c > 1.0) c = 1.0;
            if (c < -1.0) c = -1.0;
            theta1 = acos(c);
            c = (del2_sq + rcnt_sq - del1_sq)/(2*del2*rcnt);
            if (c > 1.0) c = 1.0;
            if (c < -1.0) c = -1.0;
            theta2 = acos(c);

            if (theta1 >= MY_PI2) { // pair-wise interaction between target bead and bead i1
               hbeadtube = del1;
               rscaled = hbeadtube - TubeRad_cnt;
               if (rscaled < cutoff_cnt) {
                  if (rscaled < sixth_root_sig) { // repulsive LJ core
                     r2inv = 1.0/(rscaled * rscaled);
                     r6inv = r2inv*r2inv*r2inv;
                     forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                     fpair = fscale*forcelj/rscaled;
                  }
                  else { // attractive soft tail
                     arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                     fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                  }
                  fpair1 = fpair/del1;
                  f[i1][0] -= fpair1*del1x;
                  f[i1][1] -= fpair1*del1y;
                  f[i1][2] -= fpair1*del1z;

                  f[j][0] += fpair1*del1x;
                  f[j][1] += fpair1*del1y;
                  f[j][2] += fpair1*del1z;
               }
            }
            else if (theta2 >= MY_PI2) { // pair-wise interaction between target bead and bead i2
               hbeadtube = del2;
               rscaled = hbeadtube - TubeRad_cnt;
               if (rscaled < cutoff_cnt) {
                  if (rscaled < sixth_root_sig) { // repulsive LJ core
                     r2inv = 1.0/(rscaled * rscaled);
                     r6inv = r2inv*r2inv*r2inv;
                     forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                     fpair = fscale*forcelj/rscaled;
                  }
                  else { // attractive soft tail
                     arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                     fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                  }
                  fpair2 = fpair/del2;
                  f[i2][0] -= fpair2*del2x;
                  f[i2][1] -= fpair2*del2y;
                  f[i2][2] -= fpair2*del2z;

                  f[j][0] += fpair2*del2x;
                  f[j][1] += fpair2*del2y;
                  f[j][2] += fpair2*del2z;
               }
            }
            else { // 3-body interaction amongst target bead, bead i1, and bead i2
               hbeadtube = del1 * sin(theta1); // can also be written as del2 * sin(theta2)
               rscaled = hbeadtube - TubeRad_cnt;
               if (rscaled < cutoff_cnt) {
                  if (rscaled < sixth_root_sig) { // repulsive LJ core
                     r2inv = 1.0/(rscaled * rscaled);
                     r6inv = r2inv*r2inv*r2inv;
                     forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                     fpair = fscale*forcelj/rscaled;
                  }
                  else { // attractive soft tail
                     arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                     fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                  }

                  // decide the base of the height of the triangle formed by i1, i2, and target bead
                  del1_ratio = del1*cos(theta1)/rcnt;
                  del2_ratio = 1 - del1_ratio;
                  x3tmp = x1tmp + del1_ratio * xcnt;
                  y3tmp = y1tmp + del1_ratio * ycnt;
                  z3tmp = z1tmp + del1_ratio * zcnt;

                  // vector pointing from base to target bead
                  del3x = x0tmp - x3tmp;
                  del3y = y0tmp - y3tmp;
                  del3z = z0tmp - z3tmp;
                  del3_sq = del3x*del3x + del3y*del3y + del3z*del3z;
                  del3 = sqrt(del3_sq);

                  // use lever rule to distribute force to i1 and i2 beads
                  fpair1 = fpair/del3*del2_ratio;
                  f[i1][0] -= fpair1*del3x;
                  f[i1][1] -= fpair1*del3y;
                  f[i1][2] -= fpair1*del3z;

                  fpair2 = fpair/del3*del1_ratio;
                  f[i2][0] -= fpair2*del3x;
                  f[i2][1] -= fpair2*del3y;
                  f[i2][2] -= fpair2*del3z;

                  f[j][0] += fpair*del3x/del3;
                  f[j][1] += fpair*del3y/del3;
                  f[j][2] += fpair*del3z/del3;
               }
            }
         }

    } // end of loop over neighbors of i2

  } // end of loop over all bonds

  // delete[] force_comput_flag;

  // correction is needed since some CNT beads are double-counted for bead-tube interactions.

  // anglelist is the information of all angles owned by the local processor.

  // If newton_bond is ON, then an angle is stored once on a processor where i2 is local
  // in this case, i2 is always smaller than nlocal while i1 and/or i3 can be smaller or larger than nlocal
  // depending on if i1 and/or i3 are local atoms or ghost aoms on that processor.

  // if newton_bond is OFF, then a bond is stored only on one processor if i1, i2, and i3 are all less than nlocal
  // if on a processor there is an anglewith i2 < nlocal but one of i1 and i3 is nonlocal, then the same angle
  // is also stored on the processor where i1 or i3 is local. However, the order of atoms does not change.
  // All the angles are stored as (left, middle, right). The angle is stored at least once on the processor where middle is local.
  // If left and middle are local on one processor but right is not, then the angle is stored twice: on the processor
  // where left and middle are local and on the processor where right is local.
  // if right and middle are local on one processor but left is not, then the angle is stored twice: on the processor
  // where right and middle are local and on the processor where left is local.
  // if only middle is local, but left and right belong to the same processor, then the angle is stored twice: on the processor
  // where middle is local and on the processor where left and right are local.
  // if only middle is local, but left and right belong to different processors, then the angle is stored three times: on the processor
  // middle is local, on the processor where left is local, and on the processor where right is local.

  // all these guarantee that the angle interaction is correctly calculated no matter
  // newton_bond is ON or OFF.

  // loop over all angles
  for (int ii = 0; ii < nanglelist; ii++) {

      angletype = anglelist[i][3];
      if (angletype > angletypemax_cnt || angletype <= 0) continue; // CNT angle type must be from 1 to angletypemax_cnt

      i1 = anglelist[ii][0];
      i1tag = tag[i1];
      i2 = anglelist[ii][1]; // middle bead
      i2tag = tag[i2];
      i3 = anglelist[ii][2];
      i3tag = tag[i3];

      // loop over all middle beads that are local
      if (i2 < nlocal) {

         x2tmp = x[i2][0];
         y2tmp = x[i2][1];
         z2tmp = x[i2][2];

         // loop over all neighbors of i2
         CNT_neighs = CNT_firstneigh[i2];
         for (k = 0; k < CNT_numneigh[i2]; k++) {

             j = CNT_neighs[k];
             jtag = tag[j];

             force_ignore_flag = 0;
             // check if target bead and bead i1, i2, i3 are connected by bonds
             // if yes, ignore target bead in bead-tube interaction
             if (molecule[i2] == molecule[j]) {
                if (abs(jtag - i2tag) == 1 || abs(jtag - i1tag) == 1 || abs(jtag - i3tag) == 1) {
                   force_ignore_flag = 1;
                   // fprintf (screen, "Warning: Found bonded atoms in neighbor list 3: %i %i %i %i\n", i1tag, i2tag, i3tag, jtag);
                }
             }
             if (force_ignore_flag) continue; // ignore beads directly connected to i1 or i2 or i3

             x1tmp = x[i1][0];
             y1tmp = x[i1][1];
             z1tmp = x[i1][2];

             x3tmp = x[i3][0];
             y3tmp = x[i3][1];
             z3tmp = x[i3][2];

             x0tmp = x[j][0];
             y0tmp = x[j][1];
             z0tmp = x[j][2];

             // vector pointing from bead i2 to bead i1
             xcnt1 = x1tmp - x2tmp;
             ycnt1 = y1tmp - y2tmp;
             zcnt1 = z1tmp - z2tmp;
             rcnt1_sq = xcnt1*xcnt1 + ycnt1*ycnt1 + zcnt1*zcnt1;
             rcnt1 = sqrt(rcnt1_sq);

             // vector pointing from bead i2 to bead i3
             xcnt3 = x3tmp - x2tmp;
             ycnt3 = y3tmp - y2tmp;
             zcnt3 = z3tmp - z2tmp;
             rcnt3_sq = xcnt3*xcnt3 + ycnt3*ycnt3 + zcnt3*zcnt3;
             rcnt3 = sqrt(rcnt3_sq);

             // vector pointing from bead i2 to target bead 0
             del2x = x0tmp - x2tmp;
             del2y = y0tmp - y2tmp;
             del2z = z0tmp - z2tmp;
             del2_sq = del2x*del2x + del2y*del2y + del2z*del2z;
             del2 = sqrt(del2_sq);

             // vector pointing from bead i1 to target bead 0
             del1x = x0tmp - x1tmp;
             del1y = y0tmp - y1tmp;
             del1z = z0tmp - z1tmp;
             del1_sq = del1x*del1x + del1y*del1y + del1z*del1z;
             del1 = sqrt(del1_sq);

             // vector pointing from bead i3 to target bead 0
             del3x = x0tmp - x3tmp;
             del3y = y0tmp - y3tmp;
             del3z = z0tmp - z3tmp;
             del3_sq = del3x*del3x + del3y*del3y + del3z*del3z;
             del3 = sqrt(del3_sq);

             c = (del2_sq + rcnt1_sq - del1_sq)/(2*del2*rcnt1);
             if (c > 1.0) c = 1.0;
             if (c < -1.0) c = -1.0;
             theta1 = acos(c);
             c = (del2_sq + rcnt3_sq - del3_sq)/(2*del2*rcnt3);
             if (c > 1.0) c = 1.0;
             if (c < -1.0) c = -1.0;
             theta3 = acos(c);

             // force correction is needed if target bead within cutoff from bead i2
             // case 1: when either theta1 or theta3, or both, are equal or over pi/2,
             // then the pair-wise interaction between target bead and bead i2
             // is counted twice in the calculation of bead-tube interactions
             // from the two neighboring bonds i2--i1 and i2-i3
             // if target bead is within cutoff from bead i2
             if (theta1 >= MY_PI2 || theta3 >= MY_PI2) {
                rscaled = del2 - TubeRad_cnt;
                if (rscaled < cutoff_cnt) {
                   if (rscaled < sixth_root_sig) { // repulsive LJ core
                     r2inv = 1.0/(rscaled * rscaled);
                     r6inv = r2inv*r2inv*r2inv;
                     forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                     fpair = forcelj/rscaled;
                   }
                   else { // attractive soft tail
                     arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                     fpair = -eps_by_2*pi_by_cut_sig*sin(arg);
                   }
                   fpair2 = fpair/del2;
                   f[i2][0] += fpair2*del2x;
                   f[i2][1] += fpair2*del2y;
                   f[i2][2] += fpair2*del2z;

                   f[j][0] -= fpair2*del2x;
                   f[j][1] -= fpair2*del2y;
                   f[j][2] -= fpair2*del2z;
                }
             }
             else {

                rscaled1 = del2*sin(theta1) - TubeRad_cnt;
                rscaled3 = del2*sin(theta3) - TubeRad_cnt;

                if (rscaled1 < cutoff_cnt && rscaled3 < cutoff_cnt) {

                   // case 2: target bead is closer to bond i2--i1.
                   // So the tube-bead interaction betweeen bond i2--i3
                   // and target bead is substracted below
                   if (rscaled1 < rscaled3) {
                      rscaled = rscaled3;
                      if (rscaled < sixth_root_sig) { // repulsive LJ core
                         r2inv = 1.0/(rscaled * rscaled);
                         r6inv = r2inv*r2inv*r2inv;
                         forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                         fpair = fscale*forcelj/rscaled;
                      }
                      else { // attractive soft tail
                         arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                         fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                      }

                      // decide the base of the height of the triangle formed by i2, i3, and target bead
                      del2_ratio = del2*cos(theta3)/rcnt3;
                      del3_ratio = 1 - del2_ratio;
                      x4tmp = x2tmp + del2_ratio * xcnt3;
                      y4tmp = y2tmp + del2_ratio * ycnt3;
                      z4tmp = z2tmp + del2_ratio * zcnt3;

                      // vector pointing from base to target bead
                      del4x = x0tmp - x4tmp;
                      del4y = y0tmp - y4tmp;
                      del4z = z0tmp - z4tmp;
                      del4_sq = del4x*del4x + del4y*del4y + del4z*del4z;
                      del4 = sqrt(del4_sq);

                      // use lever rule to distribute force to i2 and i3 beads for correction
                      fpair2 = fpair/del4*del3_ratio;
                      f[i2][0] += fpair2*del4x;
                      f[i2][1] += fpair2*del4y;
                      f[i2][2] += fpair2*del4z;

                      fpair3 = fpair/del4*del2_ratio;
                      f[i3][0] += fpair3*del4x;
                      f[i3][1] += fpair3*del4y;
                      f[i3][2] += fpair3*del4z;

                      f[j][0] -= fpair*del4x/del4;
                      f[j][1] -= fpair*del4y/del4;
                      f[j][2] -= fpair*del4z/del4;
                   } // end of case 2
                   // case 3: target bead is closer to bond i2--i3.
                   // So the tube-bead interaction betweeen bond i2--i1
                   // and target bead is substracted below
                   else if (rscaled3 < rscaled1) {
                      rscaled = rscaled1;
                      if (rscaled < sixth_root_sig) { // repulsive LJ core
                         r2inv = 1.0/(rscaled * rscaled);
                         r6inv = r2inv*r2inv*r2inv;
                         forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                         fpair = fscale*forcelj/rscaled;
                      }
                      else { // attractive soft tail
                         arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                         fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                      }

                      // decide the base of the height of the triangle formed by i2, i1, and target bead
                      del2_ratio = del2*cos(theta1)/rcnt1;
                      del1_ratio = 1 - del2_ratio;
                      x4tmp = x2tmp + del2_ratio * xcnt1;
                      y4tmp = y2tmp + del2_ratio * ycnt1;
                      z4tmp = z2tmp + del2_ratio * zcnt1;

                      // vector pointing from base to target bead
                      del4x = x0tmp - x4tmp;
                      del4y = y0tmp - y4tmp;
                      del4z = z0tmp - z4tmp;
                      del4_sq = del4x*del4x + del4y*del4y + del4z*del4z;
                      del4 = sqrt(del4_sq);

                      // use lever rule to distribute force to i2 and i1 beads for correction
                      fpair2 = fpair/del4*del1_ratio;
                      f[i2][0] += fpair2*del4x;
                      f[i2][1] += fpair2*del4y;
                      f[i2][2] += fpair2*del4z;

                      fpair1 = fpair/del4*del2_ratio;
                      f[i1][0] += fpair1*del4x;
                      f[i1][1] += fpair1*del4y;
                      f[i1][2] += fpair1*del4z;

                      f[j][0] -= fpair*del4x/del4;
                      f[j][1] -= fpair*del4y/del4;
                      f[j][2] -= fpair*del4z/del4;
                   } // end of case 3
                   // case 4: target bead has equal distance from i2-i1 and i2-i3 bonds
                   // in this case, each interaction contributes 50%
                   else {
                      rscaled = rscaled1; // can also write as "rscaled = rscaled3" as rscaled2==rscaled3
                      if (rscaled < sixth_root_sig) { // repulsive LJ core
                         r2inv = 1.0/(rscaled * rscaled);
                         r6inv = r2inv*r2inv*r2inv;
                         forcelj = r6inv * (lj1_cnt*r6inv - lj2_cnt);
                         fpair = fscale*forcelj/rscaled;
                      }
                      else { // attractive soft tail
                         arg = pi_by_cut_sig*(rscaled - sixth_root_sig);
                         fpair = -fscale*eps_by_2*pi_by_cut_sig*sin(arg);
                      }

                      // correct for i2--i1 bond
                      // decide the base of the height of the triangle formed by i2, i1, and target bead
                      del2_ratio = del2*cos(theta1)/rcnt1;
                      del1_ratio = 1 - del2_ratio;
                      x4tmp = x2tmp + del2_ratio * xcnt1;
                      y4tmp = y2tmp + del2_ratio * ycnt1;
                      z4tmp = z2tmp + del2_ratio * zcnt1;

                      // vector pointing from base to target bead
                      del4x = x0tmp - x4tmp;
                      del4y = y0tmp - y4tmp;
                      del4z = z0tmp - z4tmp;
                      del4_sq = del4x*del4x + del4y*del4y + del4z*del4z;
                      del4 = sqrt(del4_sq);

                      // use lever rule to distribute force to i2 and i1 beads for correction
                      fpair2 = 0.5*fpair/del4*del1_ratio;
                      f[i2][0] += fpair2*del4x;
                      f[i2][1] += fpair2*del4y;
                      f[i2][2] += fpair2*del4z;

                      fpair1 = 0.5*fpair/del4*del2_ratio;
                      f[i1][0] += fpair1*del4x;
                      f[i1][1] += fpair1*del4y;
                      f[i1][2] += fpair1*del4z;

                      f[j][0] -= 0.5*fpair*del4x/del4;
                      f[j][1] -= 0.5*fpair*del4y/del4;
                      f[j][2] -= 0.5*fpair*del4z/del4;

                      // correct for i2--i3 bond
                      // decide the base of the height of the triangle formed by i2, i3, and target bead
                      del2_ratio = del2*cos(theta3)/rcnt3;
                      del3_ratio = 1 - del2_ratio;
                      x4tmp = x2tmp + del2_ratio * xcnt3;
                      y4tmp = y2tmp + del2_ratio * ycnt3;
                      z4tmp = z2tmp + del2_ratio * zcnt3;

                      // vector pointing from base to target bead
                      del4x = x0tmp - x4tmp;
                      del4y = y0tmp - y4tmp;
                      del4z = z0tmp - z4tmp;
                      del4_sq = del4x*del4x + del4y*del4y + del4z*del4z;
                      del4 = sqrt(del4_sq);

                      // use lever rule to distribute force to i2 and i3 beads for correction
                      fpair2 = 0.5*fpair/del4*del3_ratio;
                      f[i2][0] += fpair2*del4x;
                      f[i2][1] += fpair2*del4y;
                      f[i2][2] += fpair2*del4z;

                      fpair3 = 0.5*fpair/del4*del2_ratio;
                      f[i3][0] += fpair3*del4x;
                      f[i3][1] += fpair3*del4y;
                      f[i3][2] += fpair3*del4z;

                      f[j][0] -= 0.5*fpair*del4x/del4;
                      f[j][1] -= 0.5*fpair*del4y/del4;
                      f[j][2] -= 0.5*fpair*del4z/del4;
                   } // end of case 4

                } // end of target beads within cutoff

             } // end of force correction

         }  // end loop of all neighbors of i2

    } // end loop of all middle beads that are local

  } // end loop of all angles

  // if (vflag_fdotr) virial_fdotr_compute();

}


/* ----------------------------------------------------------------------
   create CNT neighbor list from main neighbor list
   CNT neighbor list stores neighbors of ghost atoms as well
   code adapted from pair_airebo.cpp
------------------------------------------------------------------------- */

void PairBeadTube::CNT_neigh()
{
  int i,j,ii,jj,m,n,allnum,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,dS;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;

  if (atom->nmax > maxlocal) {
     maxlocal = atom->nmax;
     memory->destroy(CNT_numneigh);
     memory->sfree(CNT_firstneigh);
     memory->create(CNT_numneigh,maxlocal,"CNT:numneigh");
     CNT_firstneigh = (int **) memory->smalloc(maxlocal*sizeof(int *),
                                               "CNT:firstneigh");
  }

  allnum = list->inum + list->gnum; // number of owned and ghost atoms
  // inum = list->inum; // the length of the neighborlist list = nlocal
  ilist = list->ilist; // list of "i" atoms for which neighbor lists exist
  numneigh = list->numneigh; // the length of each neighborlist
  firstneigh = list->firstneigh; // pointer to the neighborlist of "i"

  // store all CNT neighs of owned and ghost atoms
  // scan full neighbor list of I

  ipage->reset();

  // fprintf(screen, "HELLO WORLD FROM initialization6\n");

  double c, blen = 0;
  for (m = 1; m<= bondtypemax_cnt; m++) {
       c = force->bond->equilibrium_distance(m);
       if (c > blen) blen = c;
  }
  double cnt_neigh_cutoff_sq = 1.8*(pow(0.5*blen, 2.0)+pow(TubeRad_cnt + cutoff_cnt, 2.0));

  // loop over all owned and ghost atoms
  for (ii = 0; ii < allnum; ii++) {
      i = ilist[ii];

      n = 0;
      neighptr = ipage->vget();

      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      // itype = map[type[i]];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          // jtype = map[type[j]];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;

          if (rsq < cnt_neigh_cutoff_sq) {
             neighptr[n++] = j;
          }
      }

      CNT_firstneigh[i] = neighptr;
      CNT_numneigh[i] = n;

      ipage->vgot(n);
      if (ipage->status())
         error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairBeadTube::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(rad,n+1,n+1,"pair:rad");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBeadTube::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global =force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs

Need to include CNT radius, so number of input params should be 5 or 6
input line should read:

pair_coeff atomtype1 atomtype2 epsilon sigma CNT_radius cutoff atomtype_max bondtype_max angletype_max
------------------------------------------------------------------------- */

void PairBeadTube::coeff(int narg, char **arg)
{
  if (narg < 8 || narg > 9)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  epsilon_cnt = force->numeric(FLERR, arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  sigma_cnt = force->numeric(FLERR,arg[3]);

  //CNT radius, type assignment
  double rad_one = force->numeric(FLERR,arg[4]);
  TubeRad_cnt = force->numeric(FLERR,arg[4]);

  // use global cutoff to construct neighbor-list
  double cut_one = cut_global;
  // double cut_one = force->numeric(FLERR,arg[5]);
  cutoff_cnt = force->numeric(FLERR,arg[5]);

  atomtypemax_cnt = 1;
  atomtypemax_cnt = force->numeric(FLERR,arg[6]);
  bondtypemax_cnt = 1;
  bondtypemax_cnt = force->numeric(FLERR,arg[7]);
  angletypemax_cnt = 1;
  angletypemax_cnt = force->numeric(FLERR,arg[8]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
          epsilon[i][j] = epsilon_one;
          sigma[i][j] = sigma_one;
          rad[i][j] = rad_one;
          cut[i][j] = cut_one;
          setflag[i][j] = 1;
          count++;
      }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairBeadTube::init_style()
{
  // request neighbor lists for both owned and ghost atoms

  int irequest;

  if (atom->tag_enable == 0)
     error->all(FLERR,"Pair style BeadTube requires atom IDs");
  if (force->newton_pair == 0)
     error->all(FLERR,"Pair style BeadTube requires newton pair ON");
  if (atom->molecule_flag == 0)
     error->all(FLERR,"Pair style BeadTube requires molecule IDs");
  if (force->special_lj[2] == 0.0 || force->special_coul[2] == 0.0 ||
      force->special_lj[3] == 0.0 || force->special_coul[3] == 0.0) {
      error->all(FLERR,"Pair style BeadTube requires 1-3 and 1-4 bonds turned ON in special_bond command");
  }

  irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;

  // local CNT neighbor list
  // create pages if first time or if neighbor pgsize/oneatom has changed

  int create = 0;
  if (ipage == NULL) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete [] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;

    int nmypage= comm->nthreads;
    ipage = new MyPage<int>[nmypage];
    for (int i = 0; i < nmypage; i++)
         ipage[i].init(oneatom,pgsize,PGDELTA);
  }

}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairBeadTube::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBeadTube::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    rad[i][j] = mix_distance(rad[i][i],rad[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBeadTube::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&rad[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBeadTube::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&rad[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rad[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBeadTube::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBeadTube::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairBeadTube::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %d %d %d\n",i,epsilon[i][i],sigma[i][i],rad[i][i],cutoff_cnt,atomtypemax_cnt,
    bondtypemax_cnt,angletypemax_cnt);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairBeadTube::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %d %d %d\n",i,j,epsilon[i][j],sigma[i][j],rad[i][i],cutoff_cnt,atomtypemax_cnt,
    bondtypemax_cnt,angletypemax_cnt);
}

/* ----------------------------------------------------------------------
   extract protected data from object
------------------------------------------------------------------------- */

void *PairBeadTube::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  if (strcmp(str,"rad") == 0) return (void *) rad;
  return NULL;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairBeadTube::memory_usage()
{
  double bytes = 0.0;
  bytes += maxlocal * sizeof(int);
  bytes += maxlocal * sizeof(int *);

  for (int i = 0; i < comm->nthreads; i++)
    bytes += ipage[i].size();

  bytes += 2*maxlocal * sizeof(double);
  return bytes;
}





