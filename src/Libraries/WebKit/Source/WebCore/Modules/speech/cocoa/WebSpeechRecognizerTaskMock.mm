/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#import "WebSpeechRecognizerTaskMock.h"

#if HAVE(SPEECHRECOGNIZER)

NS_ASSUME_NONNULL_BEGIN

@implementation WebSpeechRecognizerTaskMock

- (instancetype)initWithIdentifier:(WebCore::SpeechRecognitionConnectionClientIdentifier)identifier locale:(NSString*)localeIdentifier doMultipleRecognitions:(BOOL)continuous reportInterimResults:(BOOL)interimResults maxAlternatives:(unsigned long)alternatives delegateCallback:(void(^)(const WebCore::SpeechRecognitionUpdate&))callback
{
    UNUSED_PARAM(localeIdentifier);
    UNUSED_PARAM(interimResults);
    UNUSED_PARAM(alternatives);

    if (!(self = [super init]))
        return nil;

    _doMultipleRecognitions = continuous;
    _identifier = identifier;
    _delegateCallback = callback;
    _completed = false;
    _hasSentSpeechStart = false;
    _hasSentSpeechEnd = false;

    return self;
}

- (void)audioSamplesAvailable:(CMSampleBufferRef)sampleBuffer
{
    UNUSED_PARAM(sampleBuffer);
    
    if (!_hasSentSpeechStart) {
        _hasSentSpeechStart = true;
        _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::SpeechStart));
    }

    // Fake some recognition results.
    WebCore::SpeechRecognitionAlternativeData alternative { "Test"_s, 1.0 };
    _delegateCallback(WebCore::SpeechRecognitionUpdate::createResult(*_identifier, { WebCore::SpeechRecognitionResultData { { WTFMove(alternative) }, true } }));

    if (!_doMultipleRecognitions)
        [self abort];
}

- (void)abort
{
    if (_completed)
        return;
    _completed = true;

    if (!_hasSentSpeechEnd && _hasSentSpeechStart) {
        _hasSentSpeechEnd = true;
        _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::SpeechEnd));
    }

    _delegateCallback(WebCore::SpeechRecognitionUpdate::create(*_identifier, WebCore::SpeechRecognitionUpdateType::End));
}

- (void)stop
{
    [self abort];
}

@end

NS_ASSUME_NONNULL_END

#endif
