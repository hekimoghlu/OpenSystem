/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#import "config.h"
#import "WebSpeechRecognizerTask.h"

#if HAVE(SPEECHRECOGNIZER)

#import <pal/spi/cocoa/SpeechSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/WeakObjCPtr.h>

#import <pal/cocoa/SpeechSoftLink.h>

// Set the maximum duration to be an hour; we can adjust this if needed.
static constexpr size_t maximumRecognitionDuration = 60 * 60;

NS_ASSUME_NONNULL_BEGIN

@interface WebSpeechRecognizerTaskImpl : NSObject<SFSpeechRecognitionTaskDelegate, SFSpeechRecognizerDelegate> {
@private
    Markable<WebCore::SpeechRecognitionConnectionClientIdentifier> _identifier;
    BlockPtr<void(const WebCore::SpeechRecognitionUpdate&)> _delegateCallback;
    bool _doMultipleRecognitions;
    uint64_t _maxAlternatives;
    RetainPtr<SFSpeechRecognizer> _recognizer;
    RetainPtr<SFSpeechAudioBufferRecognitionRequest> _request;
    WeakObjCPtr<SFSpeechRecognitionTask> _task;
    bool _hasSentSpeechStart;
    bool _hasSentSpeechEnd;
    bool _hasSentEnd;
}

- (instancetype)initWithIdentifier:(WebCore::SpeechRecognitionConnectionClientIdentifier)identifier locale:(NSString*)localeIdentifier doMultipleRecognitions:(BOOL)continuous reportInterimResults:(BOOL)interimResults maxAlternatives:(unsigned long)alternatives delegateCallback:(void(^)(const WebCore::SpeechRecognitionUpdate&))callback;
- (void)callbackWithTranscriptions:(NSArray<SFTranscription *> *)transcriptions isFinal:(BOOL)isFinal;
- (void)audioSamplesAvailable:(CMSampleBufferRef)sampleBuffer;
- (void)abort;
- (void)stop;
- (void)sendSpeechStartIfNeeded;
- (void)sendSpeechEndIfNeeded;
- (void)sendEndIfNeeded;

@end

@implementation WebSpeechRecognizerTaskImpl

- (instancetype)initWithIdentifier:(WebCore::SpeechRecognitionConnectionClientIdentifier)identifier locale:(NSString*)localeIdentifier doMultipleRecognitions:(BOOL)continuous reportInterimResults:(BOOL)interimResults maxAlternatives:(unsigned long)alternatives delegateCallback:(void(^)(const WebCore::SpeechRecognitionUpdate&))callback
{
    if (!(self = [super init]))
        return nil;

    _identifier = identifier;
    _doMultipleRecognitions = continuous;
    _delegateCallback = callback;
    _hasSentSpeechStart = false;
    _hasSentSpeechEnd = false;
    _hasSentEnd = false;

    _maxAlternatives = alternatives ? alternatives : 1;

    if (![localeIdentifier length])
        _recognizer = adoptNS([PAL::allocSFSpeechRecognizerInstance() init]);
    else
        _recognizer = adoptNS([PAL::allocSFSpeechRecognizerInstance() initWithLocale:[NSLocale localeWithLocaleIdentifier:localeIdentifier]]);
    if (!_recognizer) {
        [self release];
        return nil;
    }

    if (![_recognizer isAvailable]) {
        [self release];
        return nil;
    }

    [_recognizer setDelegate:self];

    _request = adoptNS([PAL::allocSFSpeechAudioBufferRecognitionRequestInstance() init]);
    if ([_recognizer supportsOnDeviceRecognition])
        [_request setRequiresOnDeviceRecognition:YES];
    [_request setShouldReportPartialResults:interimResults];
    [_request setTaskHint:SFSpeechRecognitionTaskHintDictation];
    [_request setDetectMultipleUtterances:YES];
    [_request _setMaximumRecognitionDuration:maximumRecognitionDuration];

    _task = [_recognizer recognitionTaskWithRequest:_request.get() delegate:self];
    return self;
}

- (void)callbackWithTranscriptions:(NSArray<SFTranscription *> *)transcriptions isFinal:(BOOL)isFinal
{
    Vector<WebCore::SpeechRecognitionAlternativeData> alternatives;
    alternatives.reserveInitialCapacity(_maxAlternatives);
    for (SFTranscription* transcription in transcriptions) {
        // FIXME: <rdar://73629573> get confidence of SFTranscription when possible.
        double maxConfidence = 0.0;
        for (SFTranscriptionSegment* segment in [transcription segments]) {
            double confidence = [segment confidence];
            maxConfidence = maxConfidence < confidence ? confidence : maxConfidence;
        }
        alternatives.append(WebCore::SpeechRecognitionAlternativeData { [transcription formattedString], maxConfidence });
        if (alternatives.size() == _maxAlternatives)
            break;
    }
    _delegateCallback(WebCore::SpeechRecognitionUpdate::createResult(*_identifier, { WebCore::SpeechRecognitionResultData { WTFMove(alternatives), !!isFinal } }));
}

- (void)audioSamplesAvailable:(CMSampleBufferRef)sampleBuffer
{
    ASSERT(isMainThread());
    [_request appendAudioSampleBuffer:sampleBuffer];
}

- (void)abort
{
    if (!_task || [_task state] == SFSpeechRecognitionTaskStateCanceling)
        return;

    if ([_task state] == SFSpeechRecognitionTaskStateCompleted) {
        [self sendSpeechEndIfNeeded];
        [self sendEndIfNeeded];
        return;
    }

    [self sendSpeechEndIfNeeded];
    [_request endAudio];
    [_task cancel];
}

- (void)stop
{
    if (!_task || [_task state] == SFSpeechRecognitionTaskStateCanceling)
        return;

    if ([_task state] == SFSpeechRecognitionTaskStateCompleted) {
        [self sendSpeechEndIfNeeded];
        [self sendEndIfNeeded];
        return;
    }

    [self sendSpeechEndIfNeeded];
    [_request endAudio];
    [_task finish];
}

- (void)sendSpeechStartIfNeeded
{
    if (_hasSentSpeechStart)
        return;

    _hasSentSpeechStart = true;
    _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::SpeechStart));
}

- (void)sendSpeechEndIfNeeded
{
    if (!_hasSentSpeechStart || _hasSentSpeechEnd)
        return;

    _hasSentSpeechEnd = true;
    _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::SpeechEnd));
}

- (void)sendEndIfNeeded
{
    if (_hasSentEnd)
        return;

    _hasSentEnd = true;
    _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::End));
}

#pragma mark SFSpeechRecognizerDelegate

- (void)speechRecognizer:(SFSpeechRecognizer *)speechRecognizer availabilityDidChange:(BOOL)available
{
    ASSERT(isMainThread());

    if (available || !_task)
        return;

    auto error = WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::ServiceNotAllowed, "Speech recognition service becomes unavailable"_s };
    _delegateCallback(WebCore::SpeechRecognitionUpdate::createError(*_identifier, WTFMove(error)));
}

#pragma mark SFSpeechRecognitionTaskDelegate

- (void)speechRecognitionTask:(SFSpeechRecognitionTask *)task didHypothesizeTranscription:(SFTranscription *)transcription
{
    ASSERT(isMainThread());

    [self sendSpeechStartIfNeeded];
    [self callbackWithTranscriptions:[NSArray arrayWithObjects:transcription, nil] isFinal:NO];
}

- (void)speechRecognitionTask:(SFSpeechRecognitionTask *)task didFinishRecognition:(SFSpeechRecognitionResult *)recognitionResult
{
    ASSERT(isMainThread());

    if (task.state == SFSpeechRecognitionTaskStateCanceling || (!_doMultipleRecognitions && task.state == SFSpeechRecognitionTaskStateCompleted))
        return;

    [self callbackWithTranscriptions:recognitionResult.transcriptions isFinal:YES];

    if (!_doMultipleRecognitions)
        [self stop];
}

- (void)speechRecognitionTaskWasCancelled:(SFSpeechRecognitionTask *)task
{
    ASSERT(isMainThread());

    [self sendSpeechEndIfNeeded];
    [self sendEndIfNeeded];
}

- (void)speechRecognitionTask:(SFSpeechRecognitionTask *)task didFinishSuccessfully:(BOOL)successfully
{
    ASSERT(isMainThread());

    if (!successfully) {
        auto error = WebCore::SpeechRecognitionError { WebCore::SpeechRecognitionErrorType::Aborted, task.error.localizedDescription };
        _delegateCallback(WebCore::SpeechRecognitionUpdate::createError(*_identifier, WTFMove(error)));
    }
    
    [self sendEndIfNeeded];
}

@end

@implementation WebSpeechRecognizerTask

- (instancetype)initWithIdentifier:(WebCore::SpeechRecognitionConnectionClientIdentifier)identifier locale:(NSString*)localeIdentifier doMultipleRecognitions:(BOOL)continuous reportInterimResults:(BOOL)interimResults maxAlternatives:(unsigned long)alternatives delegateCallback:(void(^)(const WebCore::SpeechRecognitionUpdate&))callback
{
    if (!(self = [super init]))
        return nil;

    _impl = adoptNS([[WebSpeechRecognizerTaskImpl alloc] initWithIdentifier:identifier locale:localeIdentifier doMultipleRecognitions:continuous reportInterimResults:interimResults maxAlternatives:alternatives delegateCallback:callback]);

    if (!_impl) {
        [self release];
        return nil;
    }

    return self;
}

- (void)audioSamplesAvailable:(CMSampleBufferRef)sampleBuffer
{
    [_impl audioSamplesAvailable:sampleBuffer];
}

- (void)abort
{
    [_impl abort];
}

- (void)stop
{
    [_impl stop];
}

@end

NS_ASSUME_NONNULL_END

#endif
