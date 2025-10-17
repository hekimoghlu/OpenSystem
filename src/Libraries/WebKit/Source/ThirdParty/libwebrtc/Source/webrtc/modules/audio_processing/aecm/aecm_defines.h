/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AECM_AECM_DEFINES_H_
#define MODULES_AUDIO_PROCESSING_AECM_AECM_DEFINES_H_

#define AECM_DYNAMIC_Q /* Turn on/off dynamic Q-domain. */

/* Algorithm parameters */
#define FRAME_LEN 80 /* Total frame length, 10 ms. */

#define PART_LEN 64      /* Length of partition. */
#define PART_LEN_SHIFT 7 /* Length of (PART_LEN * 2) in base 2. */

#define PART_LEN1 (PART_LEN + 1)  /* Unique fft coefficients. */
#define PART_LEN2 (PART_LEN << 1) /* Length of partition * 2. */
#define PART_LEN4 (PART_LEN << 2) /* Length of partition * 4. */
#define FAR_BUF_LEN PART_LEN4     /* Length of buffers. */
#define MAX_DELAY 100

/* Counter parameters */
#define CONV_LEN 512              /* Convergence length used at startup. */
#define CONV_LEN2 (CONV_LEN << 1) /* Used at startup. */

/* Energy parameters */
#define MAX_BUF_LEN 64            /* History length of energy signals. */
#define FAR_ENERGY_MIN 1025       /* Lowest Far energy level: At least 2 */
                                  /* in energy. */
#define FAR_ENERGY_DIFF 929       /* Allowed difference between max */
                                  /* and min. */
#define ENERGY_DEV_OFFSET 0       /* The energy error offset in Q8. */
#define ENERGY_DEV_TOL 400        /* The energy estimation tolerance (Q8). */
#define FAR_ENERGY_VAD_REGION 230 /* Far VAD tolerance region. */

/* Stepsize parameters */
#define MU_MIN 10 /* Min stepsize 2^-MU_MIN (far end energy */
                  /* dependent). */
#define MU_MAX 1  /* Max stepsize 2^-MU_MAX (far end energy */
                  /* dependent). */
#define MU_DIFF 9 /* MU_MIN - MU_MAX */

/* Channel parameters */
#define MIN_MSE_COUNT 20 /* Min number of consecutive blocks with enough */
                         /* far end energy to compare channel estimates. */
#define MIN_MSE_DIFF 29  /* The ratio between adapted and stored channel to */
                         /* accept a new storage (0.8 in Q-MSE_RESOLUTION). */
#define MSE_RESOLUTION 5 /* MSE parameter resolution. */
#define RESOLUTION_CHANNEL16 12 /* W16 Channel in Q-RESOLUTION_CHANNEL16. */
#define RESOLUTION_CHANNEL32 28 /* W32 Channel in Q-RESOLUTION_CHANNEL. */
#define CHANNEL_VAD 16          /* Minimum energy in frequency band */
                                /* to update channel. */

/* Suppression gain parameters: SUPGAIN parameters in Q-(RESOLUTION_SUPGAIN). */
#define RESOLUTION_SUPGAIN 8 /* Channel in Q-(RESOLUTION_SUPGAIN). */
#define SUPGAIN_DEFAULT (1 << RESOLUTION_SUPGAIN) /* Default. */
#define SUPGAIN_ERROR_PARAM_A 3072            /* Estimation error parameter */
                                              /* (Maximum gain) (8 in Q8). */
#define SUPGAIN_ERROR_PARAM_B 1536            /* Estimation error parameter */
                                              /* (Gain before going down). */
#define SUPGAIN_ERROR_PARAM_D SUPGAIN_DEFAULT /* Estimation error parameter */
/* (Should be the same as Default) (1 in Q8). */
#define SUPGAIN_EPC_DT 200 /* SUPGAIN_ERROR_PARAM_C * ENERGY_DEV_TOL */

/* Defines for "check delay estimation" */
#define CORR_WIDTH 31 /* Number of samples to correlate over. */
#define CORR_MAX 16   /* Maximum correlation offset. */
#define CORR_MAX_BUF 63
#define CORR_DEV 4
#define CORR_MAX_LEVEL 20
#define CORR_MAX_LOW 4
#define CORR_BUF_LEN (CORR_MAX << 1) + 1
/* Note that CORR_WIDTH + 2*CORR_MAX <= MAX_BUF_LEN. */

#define ONE_Q14 (1 << 14)

/* NLP defines */
#define NLP_COMP_LOW 3277     /* 0.2 in Q14 */
#define NLP_COMP_HIGH ONE_Q14 /* 1 in Q14 */

#endif
