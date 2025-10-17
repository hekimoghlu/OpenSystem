/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
#include "evaluationmanager.h"
#include "policyengine.h"
#include <security_utilities/cfmunge.h>
#include <exception>
#include <vector>




namespace Security {
namespace CodeSigning {

#pragma mark -

static CFStringRef EvaluationTaskCreateKey(CFURLRef path, AuthorityType type)
{
    string pathString = std::to_string(type)+cfString(path);
    return makeCFString(pathString);
}

#pragma mark - EvaluationTask


//
// An evaluation task object manages the assessment - either directly, or in the
// form of waiting for another evaluation task to finish an assessment on the
// same target.
//
class EvaluationTask
{
public:
    CFURLRef path()      const { return mPath.get(); }
    AuthorityType type() const { return mType; }
    bool isSharable()    const { return mSharable; }
    void setUnsharable()       { mSharable = false; }

private:
    EvaluationTask(PolicyEngine *engine, CFURLRef path, AuthorityType type);
    virtual ~EvaluationTask();

    // Tasks cannot be copied.
    EvaluationTask(EvaluationTask const&) = delete;
    EvaluationTask& operator=(EvaluationTask const&) = delete;

    void performEvaluation(SecAssessmentFlags flags, CFDictionaryRef context);
    void waitForCompletion(SecAssessmentFlags flags, CFMutableDictionaryRef result);
    void kick();

    PolicyEngine                      *mPolicyEngine;
    AuthorityType                      mType;
    dispatch_queue_t                   mWorkQueue;
    dispatch_queue_t                   mFeedbackQueue;
    dispatch_semaphore_t               mAssessmentLock;
    dispatch_once_t                    mAssessmentKicked;
    int32_t                            mReferenceCount;
    int32_t                            mEvalCount;
    bool                               mSharable;

    CFCopyRef<CFURLRef>                mPath;
    CFCopyRef<CFMutableDictionaryRef>  mResult;
    std::vector<SecAssessmentFeedback> mFeedback;

    std::exception_ptr                 mExceptionToRethrow;

    friend class EvaluationManager;
};


EvaluationTask::EvaluationTask(PolicyEngine *engine, CFURLRef path, AuthorityType type) :
    mPolicyEngine(engine), mType(type), mAssessmentLock(dispatch_semaphore_create(0)),
    mAssessmentKicked(0), mReferenceCount(0), mEvalCount(0), mSharable(true),
    mExceptionToRethrow(0)
{
    mWorkQueue = dispatch_queue_create("EvaluationTask", 0);
    mFeedbackQueue = dispatch_queue_create("EvaluationTaskFeedback", 0);

    mPath = path;
    mResult.take(makeCFMutableDictionary());
}


EvaluationTask::~EvaluationTask()
{
    dispatch_release(mFeedbackQueue);
    dispatch_release(mWorkQueue);
    dispatch_release(mAssessmentLock);
}


void EvaluationTask::performEvaluation(SecAssessmentFlags flags, CFDictionaryRef context)
{
    bool performTheEvaluation = false;

    // each evaluation task performs at most a single evaluation
    if (OSAtomicIncrement32Barrier(&mEvalCount) == 1)
        performTheEvaluation = true;

    // define a block to run when the assessment has feedback available
    SecAssessmentFeedback relayFeedback = ^Boolean(CFStringRef type, CFDictionaryRef information) {

        __block Boolean proceed = true;
        dispatch_sync(mFeedbackQueue, ^{
            if (mFeedback.size() > 0) {
                proceed = false; // we need at least one interested party to proceed
                // forward the feedback to all registered listeners
                for (int i = 0; i < mFeedback.size(); ++i) {
                    proceed |= mFeedback[i](type, information);
                }
            }
        });
        if (!proceed)
            this->setUnsharable(); // don't share an expiring evaluation task
        return proceed;
    };


    // if the calling context has a feedback block, register it to listen to
    // our feedback relay
    dispatch_sync(mFeedbackQueue, ^{
        SecAssessmentFeedback feedback = (SecAssessmentFeedback)CFDictionaryGetValue(context, kSecAssessmentContextKeyFeedback);
        if (feedback && CFGetTypeID(feedback) == CFGetTypeID(relayFeedback))
            mFeedback.push_back(feedback);
    });

    // if we haven't already started the evaluation (we're the first interested
    // party), do it now
    if (performTheEvaluation) {
        dispatch_semaphore_t startLock = dispatch_semaphore_create(0);

        // create the assessment block
        dispatch_block_t assessmentBlock =
        dispatch_block_create_with_qos_class(DISPATCH_BLOCK_ENFORCE_QOS_CLASS, QOS_CLASS_UTILITY, 0, ^{
            // signal that the assessment is ready to start
            dispatch_semaphore_signal(startLock);

            // wait until we're permitted to start the assessment. if we're in low
            // priority mode, this will not happen until we're on AC power. if not
            // in low priority mode, we're either already free to perform the
            // assessment or we will be quite soon
            dispatch_semaphore_wait(mAssessmentLock, DISPATCH_TIME_FOREVER);

            // copy the original context into our own mutable dictionary and replace
            // (or assign) the feedback entry within it to our multi-receiver
            // feedback relay block
            CFRef<CFMutableDictionaryRef> contextOverride = makeCFMutableDictionary(context);
            CFDictionaryRemoveValue(contextOverride.get(), kSecAssessmentContextKeyFeedback);
            CFDictionaryAddValue(contextOverride.get(), kSecAssessmentContextKeyFeedback, relayFeedback);

            try {
                // perform the evaluation
                switch (mType) {
                    case kAuthorityExecute:
                        mPolicyEngine->evaluateCode(mPath.get(), kAuthorityExecute, flags, contextOverride.get(), mResult.get(), true);
                        break;
                    case kAuthorityInstall:
                        mPolicyEngine->evaluateInstall(mPath.get(), flags, contextOverride.get(), mResult.get());
                        break;
                    case kAuthorityOpenDoc:
                        mPolicyEngine->evaluateDocOpen(mPath.get(), flags, contextOverride.get(), mResult.get());
                        break;
                    default:
                        MacOSError::throwMe(errSecCSInvalidAttributeValues);
                }
            } catch(...) {
                mExceptionToRethrow = std::current_exception();
            }
            
        });
        assert(assessmentBlock != NULL);
        
        dispatch_async(mWorkQueue, assessmentBlock);
        Block_release(assessmentBlock);

        // wait for the assessment to start
        dispatch_semaphore_wait(startLock, DISPATCH_TIME_FOREVER);
        dispatch_release(startLock);
    }

    kick();
}

void EvaluationTask::kick() {
    dispatch_once(&mAssessmentKicked, ^{
        dispatch_semaphore_signal(mAssessmentLock);
    });
}

void EvaluationTask::waitForCompletion(SecAssessmentFlags flags, CFMutableDictionaryRef result)
{
    // if the caller didn't request low priority we will elevate the dispatch
    // queue priority via our wait block
    dispatch_qos_class_t qos_class = QOS_CLASS_USER_INITIATED;
    if (flags & kSecAssessmentFlagLowPriority)
        qos_class = QOS_CLASS_UTILITY;

    // wait for the assessment to complete; our wait block will queue up behind
    // the assessment and the copy its results
    dispatch_block_t wait_block = dispatch_block_create_with_qos_class
    (DISPATCH_BLOCK_ENFORCE_QOS_CLASS,
     qos_class, 0,
     ^{
         // copy the class result back to the caller
         cfDictionaryApplyBlock(mResult.get(),
                                ^(const void *key, const void *value){
                                    CFDictionaryAddValue(result, key, value);
                                });
     });
    assert(wait_block != NULL);
    dispatch_sync(mWorkQueue, wait_block);
    Block_release(wait_block);
}



#pragma mark -


static Boolean evaluationTasksAreEqual(const EvaluationTask *task1, const EvaluationTask *task2)
{
    if (!task1->isSharable() || !task2->isSharable()) return false;
    if ((task1->type() != task2->type()) ||
        (cfString(task1->path()) != cfString(task2->path())))
        return false;

    return true;
}




#pragma mark - EvaluationManager


EvaluationManager *EvaluationManager::globalManager()
{
    static EvaluationManager *singleton;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        singleton = new EvaluationManager();
    });
    return singleton;
}


EvaluationManager::EvaluationManager()
{
    static CFDictionaryValueCallBacks evalTaskValueCallbacks = kCFTypeDictionaryValueCallBacks;
    evalTaskValueCallbacks.equal = (CFDictionaryEqualCallBack)evaluationTasksAreEqual;
    evalTaskValueCallbacks.retain = NULL;
    evalTaskValueCallbacks.release = NULL;
    mCurrentEvaluations.take(
        CFDictionaryCreateMutable(NULL,
                              0,
                              &kCFTypeDictionaryKeyCallBacks,
                              &evalTaskValueCallbacks));

    mListLockQueue = dispatch_queue_create("EvaluationManagerSyncronization", 0);
}


EvaluationManager::~EvaluationManager()
{
    dispatch_release(mListLockQueue);
}


EvaluationTask *EvaluationManager::evaluationTask(PolicyEngine *engine, CFURLRef path, AuthorityType type, SecAssessmentFlags flags, CFDictionaryRef context, CFMutableDictionaryRef result)
{
    __block EvaluationTask *evalTask = NULL;

    dispatch_sync(mListLockQueue, ^{
        CFRef<CFStringRef> key = EvaluationTaskCreateKey(path, type);
        // is path already being evaluated?
        if (!(flags & kSecAssessmentFlagIgnoreActiveAssessments))
            evalTask = (EvaluationTask *)CFDictionaryGetValue(mCurrentEvaluations.get(), key.get());
        if (!evalTask) {
            // create a new task for the evaluation
            evalTask = new EvaluationTask(engine, path, type);
            if (flags & kSecAssessmentFlagIgnoreActiveAssessments)
                evalTask->setUnsharable();
            CFDictionaryAddValue(mCurrentEvaluations.get(), key.get(), evalTask);
        }
        evalTask->mReferenceCount++;
    });

    if (evalTask)
        evalTask->performEvaluation(flags, context);

    return evalTask;
}


void EvaluationManager::finalizeTask(EvaluationTask *task, SecAssessmentFlags flags, CFMutableDictionaryRef result)
{
    task->waitForCompletion(flags, result);

    std::exception_ptr pendingException = task->mExceptionToRethrow;

    removeTask(task);

    if (pendingException) std::rethrow_exception(pendingException);
}


void EvaluationManager::removeTask(EvaluationTask *task)
{
    dispatch_sync(mListLockQueue, ^{
        CFRef<CFStringRef> key = EvaluationTaskCreateKey(task->path(), task->type());
        // are we done with this evaluation task?
        if (--task->mReferenceCount == 0) {
            // yes -- remove it from our list and delete the object
            CFDictionaryRemoveValue(mCurrentEvaluations.get(), key.get());
            delete task;
        }
    });
}

void EvaluationManager::kickTask(CFStringRef key)
{
    dispatch_sync(mListLockQueue, ^{
        EvaluationTask *evalTask = (EvaluationTask*)CFDictionaryGetValue(mCurrentEvaluations.get(),
                                                                         key);
        if (evalTask != NULL) {
            evalTask->kick();
        }
    });
}

} // end namespace CodeSigning
} // end namespace Security

