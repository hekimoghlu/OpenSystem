/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef _H_EVALUATIONMANAGER
#define _H_EVALUATIONMANAGER

#include "policydb.h"
#include <security_utilities/cfutilities.h>

namespace Security {
namespace CodeSigning {


class PolicyEngine;
class EvaluationTask; /* an opaque type */

//
// EvaluationManager manages a list of concurrent evaluation tasks (each of
// which is wrapped within an EvaluationTask object).
//
class EvaluationManager
{
public:
    static EvaluationManager *globalManager();

    EvaluationTask *evaluationTask(PolicyEngine *engine, CFURLRef path, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result);
    void finalizeTask(EvaluationTask *task, SecAssessmentFlags flags, CFMutableDictionaryRef result);

    void kickTask(CFStringRef key);

private:
    CFCopyRef<CFMutableDictionaryRef> mCurrentEvaluations;

    EvaluationManager();
    ~EvaluationManager();

    void removeTask(EvaluationTask *task);

    dispatch_queue_t                  mListLockQueue;
};



} // end namespace CodeSigning
} // end namespace Security

#endif //_H_EVALUATIONMANAGER

