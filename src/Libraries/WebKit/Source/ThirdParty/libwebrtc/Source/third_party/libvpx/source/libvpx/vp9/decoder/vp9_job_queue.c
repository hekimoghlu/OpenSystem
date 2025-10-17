/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include <assert.h>
#include <string.h>

#include "vpx/vpx_integer.h"
#include "vpx_util/vpx_pthread.h"

#include "vp9/decoder/vp9_job_queue.h"

void vp9_jobq_init(JobQueueRowMt *jobq, uint8_t *buf, size_t buf_size) {
#if CONFIG_MULTITHREAD
  pthread_mutex_init(&jobq->mutex, NULL);
  pthread_cond_init(&jobq->cond, NULL);
#endif
  jobq->buf_base = buf;
  jobq->buf_wr = buf;
  jobq->buf_rd = buf;
  jobq->buf_end = buf + buf_size;
  jobq->terminate = 0;
}

void vp9_jobq_reset(JobQueueRowMt *jobq) {
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(&jobq->mutex);
#endif
  jobq->buf_wr = jobq->buf_base;
  jobq->buf_rd = jobq->buf_base;
  jobq->terminate = 0;
#if CONFIG_MULTITHREAD
  pthread_mutex_unlock(&jobq->mutex);
#endif
}

void vp9_jobq_deinit(JobQueueRowMt *jobq) {
  vp9_jobq_reset(jobq);
#if CONFIG_MULTITHREAD
  pthread_mutex_destroy(&jobq->mutex);
  pthread_cond_destroy(&jobq->cond);
#endif
}

void vp9_jobq_terminate(JobQueueRowMt *jobq) {
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(&jobq->mutex);
#endif
  jobq->terminate = 1;
#if CONFIG_MULTITHREAD
  pthread_cond_broadcast(&jobq->cond);
  pthread_mutex_unlock(&jobq->mutex);
#endif
}

int vp9_jobq_queue(JobQueueRowMt *jobq, void *job, size_t job_size) {
  int ret = 0;
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(&jobq->mutex);
#endif
  if (jobq->buf_end >= jobq->buf_wr + job_size) {
    memcpy(jobq->buf_wr, job, job_size);
    jobq->buf_wr = jobq->buf_wr + job_size;
#if CONFIG_MULTITHREAD
    pthread_cond_signal(&jobq->cond);
#endif
    ret = 0;
  } else {
    /* Wrap around case is not supported */
    assert(0);
    ret = 1;
  }
#if CONFIG_MULTITHREAD
  pthread_mutex_unlock(&jobq->mutex);
#endif
  return ret;
}

int vp9_jobq_dequeue(JobQueueRowMt *jobq, void *job, size_t job_size,
                     int blocking) {
  int ret = 0;
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(&jobq->mutex);
#endif
  if (jobq->buf_end >= jobq->buf_rd + job_size) {
    while (1) {
      if (jobq->buf_wr >= jobq->buf_rd + job_size) {
        memcpy(job, jobq->buf_rd, job_size);
        jobq->buf_rd = jobq->buf_rd + job_size;
        ret = 0;
        break;
      } else {
        /* If all the entries have been dequeued, then break and return */
        if (jobq->terminate == 1) {
          ret = 1;
          break;
        }
        if (blocking == 1) {
#if CONFIG_MULTITHREAD
          pthread_cond_wait(&jobq->cond, &jobq->mutex);
#endif
        } else {
          /* If there is no job available,
           * and this is non blocking call then return fail */
          ret = 1;
          break;
        }
      }
    }
  } else {
    /* Wrap around case is not supported */
    ret = 1;
  }
#if CONFIG_MULTITHREAD
  pthread_mutex_unlock(&jobq->mutex);
#endif

  return ret;
}
