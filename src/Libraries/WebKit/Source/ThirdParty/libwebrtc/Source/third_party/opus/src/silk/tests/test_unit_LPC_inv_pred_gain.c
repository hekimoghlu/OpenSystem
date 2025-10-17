/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include <stdio.h>
#include <stdlib.h>
#include "celt/stack_alloc.h"
#include "cpu_support.h"
#include "SigProc_FIX.h"

/* Computes the impulse response of the filter so we
   can catch filters that are definitely unstable. Some
   unstable filters may be classified as stable, but not
   the other way around. */
int check_stability(opus_int16 *A_Q12, int order) {
    int i;
    int j;
    int sum_a, sum_abs_a;
    double y[SILK_MAX_ORDER_LPC] = {0};
    sum_a = sum_abs_a = 0;
    for( j = 0; j < order; j++ ) {
        sum_a += A_Q12[ j ];
        sum_abs_a += silk_abs( A_Q12[ j ] );
    }
    /* Check DC stability. */
    if( sum_a >= 4096 ) {
        return 0;
    }
    /* If the sum of absolute values is less than 1, the filter
       has to be stable. */
    if( sum_abs_a < 4096 ) {
        return 1;
    }
    y[0] = 1;
    for( i = 0; i < 10000; i++ ) {
        double sum = 0;
        for( j = 0; j < order; j++ ) {
            sum += y[ j ]*A_Q12[ j ];
        }
        for( j = order - 1; j > 0; j-- ) {
            y[ j ] = y[ j - 1 ];
        }
        y[ 0 ] = sum*(1./4096);
        /* If impulse response reaches +/- 10000, the filter
           is definitely unstable. */
        if( !(y[ 0 ] < 10000 && y[ 0 ] > -10000) ) {
            return 0;
        }
        /* Test every 8 sample for low amplitude. */
        if( ( i & 0x7 ) == 0 ) {
            double amp = 0;
            for( j = 0; j < order; j++ ) {
                amp += fabs(y[j]);
            }
            if( amp < 0.00001 ) {
                return 1;
            }
        }
    }
    return 1;
}

int main(void) {
    const int arch = opus_select_arch();
    /* Set to 10000 so all branches in C function are triggered */
    const int loop_num = 10000;
    int count = 0;
    ALLOC_STACK;

    /* FIXME: Make the seed random (with option to set it explicitly)
       so we get wider coverage. */
    srand(0);

    printf("Testing silk_LPC_inverse_pred_gain() optimization ...\n");
    for( count = 0; count < loop_num; count++ ) {
        unsigned int i;
        opus_int     order;
        unsigned int shift;
        opus_int16   A_Q12[ SILK_MAX_ORDER_LPC ];
        opus_int32 gain;

        for( order = 2; order <= SILK_MAX_ORDER_LPC; order += 2 ) { /* order must be even. */
            for( shift = 0; shift < 16; shift++ ) { /* Different dynamic range. */
                for( i = 0; i < SILK_MAX_ORDER_LPC; i++ ) {
                    A_Q12[i] = ((opus_int16)rand()) >> shift;
                }
                gain = silk_LPC_inverse_pred_gain(A_Q12, order, arch);
                /* Look for filters that silk_LPC_inverse_pred_gain() thinks are
                   stable but definitely aren't. */
                if( gain != 0 && !check_stability(A_Q12, order) ) {
                    fprintf(stderr, "**Loop %4d failed!**\n", count);
                    return 1;
                }
            }
        }
        if( !(count % 500) ) {
            printf("Loop %4d passed\n", count);
        }
    }
    printf("silk_LPC_inverse_pred_gain() optimization passed\n");
    return 0;
}
