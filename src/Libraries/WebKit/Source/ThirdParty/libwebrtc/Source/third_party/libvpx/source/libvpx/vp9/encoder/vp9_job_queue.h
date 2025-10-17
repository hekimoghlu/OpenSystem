/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_JOB_QUEUE_H_
#define VPX_VP9_ENCODER_VP9_JOB_QUEUE_H_

typedef enum {
  FIRST_PASS_JOB,
  ENCODE_JOB,
  ARNR_JOB,
  NUM_JOB_TYPES,
} JOB_TYPE;

// Encode job parameters
typedef struct {
  int vert_unit_row_num;  // Index of the vertical unit row
  int tile_col_id;        // tile col id within a tile
  int tile_row_id;        // tile col id within a tile
} JobNode;

// Job queue element parameters
typedef struct {
  // Pointer to the next link in the job queue
  void *next;

  // Job information context of the module
  JobNode job_info;
} JobQueue;

// Job queue handle
typedef struct {
  // Pointer to the next link in the job queue
  void *next;

  // Counter to store the number of jobs picked up for processing
  int num_jobs_acquired;
} JobQueueHandle;

#endif  // VPX_VP9_ENCODER_VP9_JOB_QUEUE_H_
