/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#ifndef VPX_VP9_DECODER_VP9_JOB_QUEUE_H_
#define VPX_VP9_DECODER_VP9_JOB_QUEUE_H_

#include "vpx_util/vpx_pthread.h"

typedef struct {
  // Pointer to buffer base which contains the jobs
  uint8_t *buf_base;

  // Pointer to current address where new job can be added
  uint8_t *volatile buf_wr;

  // Pointer to current address from where next job can be obtained
  uint8_t *volatile buf_rd;

  // Pointer to end of job buffer
  uint8_t *buf_end;

  int terminate;

#if CONFIG_MULTITHREAD
  pthread_mutex_t mutex;
  pthread_cond_t cond;
#endif
} JobQueueRowMt;

void vp9_jobq_init(JobQueueRowMt *jobq, uint8_t *buf, size_t buf_size);
void vp9_jobq_reset(JobQueueRowMt *jobq);
void vp9_jobq_deinit(JobQueueRowMt *jobq);
void vp9_jobq_terminate(JobQueueRowMt *jobq);
int vp9_jobq_queue(JobQueueRowMt *jobq, void *job, size_t job_size);
int vp9_jobq_dequeue(JobQueueRowMt *jobq, void *job, size_t job_size,
                     int blocking);

#endif  // VPX_VP9_DECODER_VP9_JOB_QUEUE_H_
