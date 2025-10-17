/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#import "RTCAudioSession+Private.h"
#import "RTCAudioSessionConfiguration.h"

#import "base/RTCLogging.h"

@implementation RTCAudioSession (Configuration)

- (BOOL)setConfiguration:(RTCAudioSessionConfiguration *)configuration
                   error:(NSError **)outError {
  return [self setConfiguration:configuration
                         active:NO
                shouldSetActive:NO
                          error:outError];
}

- (BOOL)setConfiguration:(RTCAudioSessionConfiguration *)configuration
                  active:(BOOL)active
                   error:(NSError **)outError {
  return [self setConfiguration:configuration
                         active:active
                shouldSetActive:YES
                          error:outError];
}

#pragma mark - Private

- (BOOL)setConfiguration:(RTCAudioSessionConfiguration *)configuration
                  active:(BOOL)active
         shouldSetActive:(BOOL)shouldSetActive
                   error:(NSError **)outError {
  NSParameterAssert(configuration);
  if (outError) {
    *outError = nil;
  }
  if (![self checkLock:outError]) {
    return NO;
  }

  // Provide an error even if there isn't one so we can log it. We will not
  // return immediately on error in this function and instead try to set
  // everything we can.
  NSError *error = nil;

  if (self.category != configuration.category ||
      self.categoryOptions != configuration.categoryOptions) {
    NSError *categoryError = nil;
    if (![self setCategory:configuration.category
               withOptions:configuration.categoryOptions
                     error:&categoryError]) {
      RTCLogError(@"Failed to set category: %@",
                  categoryError.localizedDescription);
      error = categoryError;
    } else {
      RTCLog(@"Set category to: %@", configuration.category);
    }
  }

  if (self.mode != configuration.mode) {
    NSError *modeError = nil;
    if (![self setMode:configuration.mode error:&modeError]) {
      RTCLogError(@"Failed to set mode: %@",
                  modeError.localizedDescription);
      error = modeError;
    } else {
      RTCLog(@"Set mode to: %@", configuration.mode);
    }
  }

  // Sometimes category options don't stick after setting mode.
  if (self.categoryOptions != configuration.categoryOptions) {
    NSError *categoryError = nil;
    if (![self setCategory:configuration.category
               withOptions:configuration.categoryOptions
                     error:&categoryError]) {
      RTCLogError(@"Failed to set category options: %@",
                  categoryError.localizedDescription);
      error = categoryError;
    } else {
      RTCLog(@"Set category options to: %ld",
             (long)configuration.categoryOptions);
    }
  }

  if (self.preferredSampleRate != configuration.sampleRate) {
    NSError *sampleRateError = nil;
    if (![self setPreferredSampleRate:configuration.sampleRate
                                error:&sampleRateError]) {
      RTCLogError(@"Failed to set preferred sample rate: %@",
                  sampleRateError.localizedDescription);
      if (!self.ignoresPreferredAttributeConfigurationErrors) {
        error = sampleRateError;
      }
    } else {
      RTCLog(@"Set preferred sample rate to: %.2f",
             configuration.sampleRate);
    }
  }

  if (self.preferredIOBufferDuration != configuration.ioBufferDuration) {
    NSError *bufferDurationError = nil;
    if (![self setPreferredIOBufferDuration:configuration.ioBufferDuration
                                      error:&bufferDurationError]) {
      RTCLogError(@"Failed to set preferred IO buffer duration: %@",
                  bufferDurationError.localizedDescription);
      if (!self.ignoresPreferredAttributeConfigurationErrors) {
        error = bufferDurationError;
      }
    } else {
      RTCLog(@"Set preferred IO buffer duration to: %f",
             configuration.ioBufferDuration);
    }
  }

  if (shouldSetActive) {
    NSError *activeError = nil;
    if (![self setActive:active error:&activeError]) {
      RTCLogError(@"Failed to setActive to %d: %@",
                  active, activeError.localizedDescription);
      error = activeError;
    }
  }

  if (self.isActive &&
      // TODO(tkchin): Figure out which category/mode numChannels is valid for.
      [self.mode isEqualToString:AVAudioSessionModeVoiceChat]) {
    // Try to set the preferred number of hardware audio channels. These calls
    // must be done after setting the audio sessionâ€™s category and mode and
    // activating the session.
    NSInteger inputNumberOfChannels = configuration.inputNumberOfChannels;
    if (self.inputNumberOfChannels != inputNumberOfChannels) {
      NSError *inputChannelsError = nil;
      if (![self setPreferredInputNumberOfChannels:inputNumberOfChannels
                                             error:&inputChannelsError]) {
       RTCLogError(@"Failed to set preferred input number of channels: %@",
                   inputChannelsError.localizedDescription);
       if (!self.ignoresPreferredAttributeConfigurationErrors) {
         error = inputChannelsError;
       }
      } else {
        RTCLog(@"Set input number of channels to: %ld",
               (long)inputNumberOfChannels);
      }
    }
    NSInteger outputNumberOfChannels = configuration.outputNumberOfChannels;
    if (self.outputNumberOfChannels != outputNumberOfChannels) {
      NSError *outputChannelsError = nil;
      if (![self setPreferredOutputNumberOfChannels:outputNumberOfChannels
                                              error:&outputChannelsError]) {
        RTCLogError(@"Failed to set preferred output number of channels: %@",
                    outputChannelsError.localizedDescription);
        if (!self.ignoresPreferredAttributeConfigurationErrors) {
          error = outputChannelsError;
        }
      } else {
        RTCLog(@"Set output number of channels to: %ld",
               (long)outputNumberOfChannels);
      }
    }
  }

  if (outError) {
    *outError = error;
  }

  return error == nil;
}

@end
