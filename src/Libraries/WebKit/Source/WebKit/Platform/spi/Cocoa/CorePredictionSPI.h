/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#pragma once

#if HAVE(CORE_PREDICTION)

#if USE(APPLE_INTERNAL_SDK)

#import <CorePrediction/svm.h>

#else

struct svm_node
{
    int index;
    double value;
};

#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct svm_node svm_node;

struct svm_model *svm_load_model(const char *model_file_name);
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);

#ifdef __cplusplus
}
#endif

#endif
