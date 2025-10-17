/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "debug.h"

#if SILK_DEBUG || SILK_TIC_TOC
#include "SigProc_FIX.h"
#endif

#if SILK_TIC_TOC

#if (defined(_WIN32) || defined(_WINCE))
#include <windows.h>    /* timer */
#else   /* Linux or Mac*/
#include <sys/time.h>
#endif

#ifdef _WIN32
unsigned long silk_GetHighResolutionTime(void) /* O  time in usec*/
{
    /* Returns a time counter in microsec   */
    /* the resolution is platform dependent */
    /* but is typically 1.62 us resolution  */
    LARGE_INTEGER lpPerformanceCount;
    LARGE_INTEGER lpFrequency;
    QueryPerformanceCounter(&lpPerformanceCount);
    QueryPerformanceFrequency(&lpFrequency);
    return (unsigned long)((1000000*(lpPerformanceCount.QuadPart)) / lpFrequency.QuadPart);
}
#else   /* Linux or Mac*/
unsigned long GetHighResolutionTime(void) /* O  time in usec*/
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return((tv.tv_sec*1000000)+(tv.tv_usec));
}
#endif

int           silk_Timer_nTimers = 0;
int           silk_Timer_depth_ctr = 0;
char          silk_Timer_tags[silk_NUM_TIMERS_MAX][silk_NUM_TIMERS_MAX_TAG_LEN];
#ifdef _WIN32
LARGE_INTEGER silk_Timer_start[silk_NUM_TIMERS_MAX];
#else
unsigned long silk_Timer_start[silk_NUM_TIMERS_MAX];
#endif
unsigned int  silk_Timer_cnt[silk_NUM_TIMERS_MAX];
opus_int64     silk_Timer_min[silk_NUM_TIMERS_MAX];
opus_int64     silk_Timer_sum[silk_NUM_TIMERS_MAX];
opus_int64     silk_Timer_max[silk_NUM_TIMERS_MAX];
opus_int64     silk_Timer_depth[silk_NUM_TIMERS_MAX];

#ifdef _WIN32
void silk_TimerSave(char *file_name)
{
    if( silk_Timer_nTimers > 0 )
    {
        int k;
        FILE *fp;
        LARGE_INTEGER lpFrequency;
        LARGE_INTEGER lpPerformanceCount1, lpPerformanceCount2;
        int del = 0x7FFFFFFF;
        double avg, sum_avg;
        /* estimate overhead of calling performance counters */
        for( k = 0; k < 1000; k++ ) {
            QueryPerformanceCounter(&lpPerformanceCount1);
            QueryPerformanceCounter(&lpPerformanceCount2);
            lpPerformanceCount2.QuadPart -= lpPerformanceCount1.QuadPart;
            if( (int)lpPerformanceCount2.LowPart < del )
                del = lpPerformanceCount2.LowPart;
        }
        QueryPerformanceFrequency(&lpFrequency);
        /* print results to file */
        sum_avg = 0.0f;
        for( k = 0; k < silk_Timer_nTimers; k++ ) {
            if (silk_Timer_depth[k] == 0) {
                sum_avg += (1e6 * silk_Timer_sum[k] / silk_Timer_cnt[k] - del) / lpFrequency.QuadPart * silk_Timer_cnt[k];
            }
        }
        fp = fopen(file_name, "w");
        fprintf(fp, "                                min         avg     %%         max      count\n");
        for( k = 0; k < silk_Timer_nTimers; k++ ) {
            if (silk_Timer_depth[k] == 0) {
                fprintf(fp, "%-28s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 1) {
                fprintf(fp, " %-27s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 2) {
                fprintf(fp, "  %-26s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 3) {
                fprintf(fp, "   %-25s", silk_Timer_tags[k]);
            } else {
                fprintf(fp, "    %-24s", silk_Timer_tags[k]);
            }
            avg = (1e6 * silk_Timer_sum[k] / silk_Timer_cnt[k] - del) / lpFrequency.QuadPart;
            fprintf(fp, "%8.2f", (1e6 * (silk_max_64(silk_Timer_min[k] - del, 0))) / lpFrequency.QuadPart);
            fprintf(fp, "%12.2f %6.2f", avg, 100.0 * avg / sum_avg * silk_Timer_cnt[k]);
            fprintf(fp, "%12.2f", (1e6 * (silk_max_64(silk_Timer_max[k] - del, 0))) / lpFrequency.QuadPart);
            fprintf(fp, "%10d\n", silk_Timer_cnt[k]);
        }
        fprintf(fp, "                                microseconds\n");
        fclose(fp);
    }
}
#else
void silk_TimerSave(char *file_name)
{
    if( silk_Timer_nTimers > 0 )
    {
        int k;
        FILE *fp;
        /* print results to file */
        fp = fopen(file_name, "w");
        fprintf(fp, "                                min         avg         max      count\n");
        for( k = 0; k < silk_Timer_nTimers; k++ )
        {
            if (silk_Timer_depth[k] == 0) {
                fprintf(fp, "%-28s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 1) {
                fprintf(fp, " %-27s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 2) {
                fprintf(fp, "  %-26s", silk_Timer_tags[k]);
            } else if (silk_Timer_depth[k] == 3) {
                fprintf(fp, "   %-25s", silk_Timer_tags[k]);
            } else {
                fprintf(fp, "    %-24s", silk_Timer_tags[k]);
            }
            fprintf(fp, "%d ", silk_Timer_min[k]);
            fprintf(fp, "%f ", (double)silk_Timer_sum[k] / (double)silk_Timer_cnt[k]);
            fprintf(fp, "%d ", silk_Timer_max[k]);
            fprintf(fp, "%10d\n", silk_Timer_cnt[k]);
        }
        fprintf(fp, "                                microseconds\n");
        fclose(fp);
    }
}
#endif

#endif /* SILK_TIC_TOC */

#if SILK_DEBUG
FILE *silk_debug_store_fp[ silk_NUM_STORES_MAX ];
int silk_debug_store_count = 0;
#endif /* SILK_DEBUG */

