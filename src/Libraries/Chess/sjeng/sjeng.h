/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef SJENG_H
#define SJENG_H

#include "config.h"
#include <ctype.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef HAVE_SYS_TIMEB_H
#include <sys/timeb.h>
#endif

#define NDEBUG 
#include <assert.h>

#define DIE (*(int *)(NULL) = 0)

/* GCP : my code uses WHITE=0 and BLACK=1 so reverse this */

#define WHITE 0
#define BLACK 1

#define ToMove (white_to_move ? 0 : 1)
#define NotToMove (white_to_move ? 1 : 0)

#define Hash(x,y) (hash ^= zobrist[(x)][(y)])

#define Crazyhouse 0
#define Bughouse 1
#define Normal 2
#define Suicide 3
#define Losers 4

#define Opening      0
#define Middlegame   1
#define Endgame      2

#define mindepth 2

/* define names for piece constants: */
#define frame   0
#define wpawn   1
#define bpawn   2
#define wknight 3
#define bknight 4
#define wking   5
#define bking   6
#define wrook   7
#define brook   8
#define wqueen  9
#define bqueen  10
#define wbishop 11
#define bbishop 12
#define npiece  13

/* result flags: */
#define no_result      0
#define stalemate      1
#define white_is_mated 2
#define black_is_mated 3
#define draw_by_fifty  4
#define draw_by_rep    5

/* arrays maybe ? */ 
#undef FASTCALC
#ifdef FASTCALC
#define rank(square) ((((square)-26)/12)+1)
#define file(square) ((((square)-26)%12)+1)
#else
#define rank(square) (rank[(square)])
#define file(square) (file[(square)])
#endif
#define diagl(square) (diagl[(square)])
#define diagr(square) (diagr[(square)])

#ifndef INPROBECODE
typedef enum {FALSE, TRUE} bool;
#endif

/* castle flags: */
#define no_castle  0
#define wck        1
#define wcq        2
#define bck        3
#define bcq        4

typedef struct {
  int from;
  int target;
  int captured;
  int promoted;	      
  int castled;
  int ep; 
} move_s;

typedef struct {
  int cap_num;
  int was_promoted;
  int epsq;
  int fifty;
} move_x;

#if defined(HAVE_SYS_TIMEB_H) && (defined(HAVE_FTIME) || defined(HAVE_GETTIMEOFDAY)) 
typedef struct timeb rtime_t;
#else
typedef time_t rtime_t;
#endif

#define STR_BUFF 256
#define MOVE_BUFF 512
#define INF 1000000
#define PV_BUFF 300

#define AddMaterial(x) Material += material[(x)]
#define RemoveMaterial(x) Material -= material[(x)]

#define UPPER 1
#define LOWER 2
#define EXACT 3
#define HMISS 4
#define DUMMY 0

#define LOSS 0
#define WIN 1
#define DRAW 2

#define max(x, y) ((x) > (y) ? (x) : (y))
#define mix(x, y) ((x) < (y) ? (x) : (y))

#endif
