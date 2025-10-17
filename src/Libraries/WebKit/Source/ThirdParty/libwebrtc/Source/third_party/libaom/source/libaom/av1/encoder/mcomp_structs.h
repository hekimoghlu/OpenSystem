/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#ifndef AOM_AV1_ENCODER_MCOMP_STRUCTS_H_
#define AOM_AV1_ENCODER_MCOMP_STRUCTS_H_

#include "av1/common/mv.h"

// The maximum number of steps in a step search given the largest
// allowed initial step
#define MAX_MVSEARCH_STEPS 11
// Max full pel mv specified in the unit of full pixel
// Enable the use of motion vector in range [-1023, 1023].
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS - 1)) - 1)
// Maximum size of the first step in full pel units
#define MAX_FIRST_STEP (1 << (MAX_MVSEARCH_STEPS - 1))
// Maximum number of neighbors to scan per iteration during
// WARPED_CAUSAL refinement
// Note: The elements of warp_search_config.neighbor_mask must be at least
// MAX_WARP_SEARCH_NEIGHBORS many bits wide. So the type may need to be
// widened if this value is increased.
#define MAX_WARP_SEARCH_NEIGHBORS 8

#define SEARCH_RANGE_8P 3
#define SEARCH_GRID_STRIDE_8P (2 * SEARCH_RANGE_8P + 1)
#define SEARCH_GRID_CENTER_8P \
  (SEARCH_RANGE_8P * SEARCH_GRID_STRIDE_8P + SEARCH_RANGE_8P)

typedef struct {
  FULLPEL_MV coord;
  int coord_offset;
} search_neighbors;
// motion search site
typedef struct search_site {
  FULLPEL_MV mv;
  int offset;
} search_site;

typedef struct search_site_config {
  search_site site[MAX_MVSEARCH_STEPS * 2][16 + 1];
  // Number of search steps.
  int num_search_steps;
  int searches_per_step[MAX_MVSEARCH_STEPS * 2];
  int radius[MAX_MVSEARCH_STEPS * 2];
  int stride;
} search_site_config;

enum {
  // Search 8-points in the radius grid around center, up to 11 search stages.
  DIAMOND = 0,
  // Search 12-points in the radius/tan_radius grid around center,
  // up to 15 search stages.
  NSTEP = 1,
  // Search 8-points in the radius grid around center, up to 16 search stages.
  NSTEP_8PT = 2,
  // Search 8-points in the radius grid around center, upto 11 search stages
  // with clamping of search radius.
  CLAMPED_DIAMOND = 3,
  // Search maximum 8-points in the radius grid around center,
  // up to 11 search stages. First stage consists of 8 search points
  // and the rest with 6 search points each in hex shape.
  HEX = 4,
  // Search maximum 8-points in the radius grid around center,
  // up to 11 search stages. First stage consists of 4 search
  // points and the rest with 8 search points each.
  BIGDIA = 5,
  // Search 8-points in the square grid around center, up to 11 search stages.
  SQUARE = 6,
  // HEX search with up to 2 stages.
  FAST_HEX = 7,
  // BIGDIA search with up to 2 stages.
  FAST_DIAMOND = 8,
  // BIGDIA search with up to 3 stages.
  FAST_BIGDIA = 9,
  // BIGDIA search with up to 1 stage.
  VFAST_DIAMOND = 10,
  // Total number of search methods.
  NUM_SEARCH_METHODS,
  // Number of distinct search methods.
  NUM_DISTINCT_SEARCH_METHODS = SQUARE + 1,
} UENUM1BYTE(SEARCH_METHODS);

typedef struct warp_search_config {
  int num_neighbors;
  MV neighbors[MAX_WARP_SEARCH_NEIGHBORS];
  // Bitmask which is used to prune the search neighbors at one iteration
  // based on which direction we chose in the previous iteration.
  // See comments in av1_refine_warped_mv for details.
  uint8_t neighbor_mask[MAX_WARP_SEARCH_NEIGHBORS];
} warp_search_config;

// Methods for refining WARPED_CAUSAL motion vectors
enum {
  // Search 4 adjacent points in a diamond shape at each iteration
  WARP_SEARCH_DIAMOND,
  // Search 8 adjacent points in a square at each iteration
  WARP_SEARCH_SQUARE,
  WARP_SEARCH_METHODS
} UENUM1BYTE(WARP_SEARCH_METHOD);

#endif  // AOM_AV1_ENCODER_MCOMP_STRUCTS_H_
