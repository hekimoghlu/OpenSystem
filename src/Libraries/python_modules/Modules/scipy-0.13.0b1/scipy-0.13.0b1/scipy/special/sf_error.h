/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#ifndef SF_ERROR_H_
#define SF_ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SF_ERROR_OK = 0,      /* no error */
    SF_ERROR_SINGULAR,    /* singularity encountered */
    SF_ERROR_UNDERFLOW,   /* floating point underflow */
    SF_ERROR_OVERFLOW,    /* floating point overflow */
    SF_ERROR_SLOW,        /* too many iterations required */
    SF_ERROR_LOSS,        /* loss of precision */
    SF_ERROR_NO_RESULT,   /* no result obtained */
    SF_ERROR_DOMAIN,      /* out of domain */
    SF_ERROR_ARG,         /* invalid input parameter */
    SF_ERROR_OTHER,       /* unclassified error */
    SF_ERROR__LAST         
} sf_error_t;

extern const char *sf_error_messages[];
void sf_error(char *func_name, sf_error_t code, char *fmt, ...);
void sf_error_check_fpe(char *func_name);
int sf_error_set_print(int flag);
int sf_error_get_print();

#ifdef __cplusplus
}
#endif

#endif /* SF_ERROR_H_ */
