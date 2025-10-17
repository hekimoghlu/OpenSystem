/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#import <Foundation/Foundation.h>
#import <HID/NSError+IOReturn.h>
#import <IOKit/IOReturn.h>

@implementation NSError (IOReturn)

+ (NSError *)errorWithIOReturn:(IOReturn)code
{
    const NSDictionary *errors = @{
        @(kIOReturnSuccess):            @"success",
        @(kIOReturnError):              @"general error",
        @(kIOReturnNoMemory):           @"memory allocation error",
        @(kIOReturnNoResources):        @"resource shortage",
        @(kIOReturnIPCError):           @"Mach IPC failure",
        @(kIOReturnNoDevice):           @"no such device",
        @(kIOReturnNotPrivileged):      @"privilege violation",
        @(kIOReturnBadArgument):        @"invalid argument",
        @(kIOReturnLockedRead):         @"device is read locked",
        @(kIOReturnLockedWrite):        @"device is write locked",
        @(kIOReturnExclusiveAccess):    @"device is exclusive access",
        @(kIOReturnBadMessageID):       @"bad IPC message ID",
        @(kIOReturnUnsupported):        @"unsupported function",
        @(kIOReturnVMError):            @"virtual memory error",
        @(kIOReturnInternalError):      @"internal driver error",
        @(kIOReturnIOError):            @"I/O error",
        @(kIOReturnCannotLock):         @"cannot acquire lock",
        @(kIOReturnNotOpen):            @"device is not open",
        @(kIOReturnNotReadable):        @"device is not readable",
        @(kIOReturnNotWritable):        @"device is not writeable",
        @(kIOReturnNotAligned):         @"alignment error",
        @(kIOReturnBadMedia):           @"media error",
        @(kIOReturnStillOpen):          @"device is still open",
        @(kIOReturnRLDError):           @"rld failure",
        @(kIOReturnDMAError):           @"DMA failure",
        @(kIOReturnBusy):               @"device is busy",
        @(kIOReturnTimeout):            @"I/O timeout",
        @(kIOReturnOffline):            @"device is offline",
        @(kIOReturnNotReady):           @"device is not ready",
        @(kIOReturnNotAttached):        @"device/channel is not attached",
        @(kIOReturnNoChannels):         @"no DMA channels available",
        @(kIOReturnNoSpace):            @"no space for data",
        @(kIOReturnPortExists):         @"device port already exists",
        @(kIOReturnCannotWire):         @"cannot wire physical memory",
        @(kIOReturnNoInterrupt):        @"no interrupt attached",
        @(kIOReturnNoFrames):           @"no DMA frames enqueued",
        @(kIOReturnMessageTooLarge):    @"message is too large",
        @(kIOReturnNotPermitted):       @"operation is not permitted",
        @(kIOReturnNoPower):            @"device is without power",
        @(kIOReturnNoMedia):            @"media is not present",
        @(kIOReturnUnformattedMedia):   @"media is not formatted",
        @(kIOReturnUnsupportedMode):    @"unsupported mode",
        @(kIOReturnUnderrun):           @"data underrun",
        @(kIOReturnOverrun):            @"data overrun",
        @(kIOReturnDeviceError):        @"device error",
        @(kIOReturnNoCompletion):       @"no completion routine",
        @(kIOReturnAborted):            @"operation was aborted",
        @(kIOReturnNoBandwidth):        @"bus bandwidth would be exceeded",
        @(kIOReturnNotResponding):      @"device is not responding",
        @(kIOReturnInvalid):            @"unanticipated driver error",
      };
    
    NSString *errorText = errors[@(code)];
    
    if (!errorText) {
        errorText = [NSString stringWithFormat:@"0x%x unknown", code];
    }
    
    NSDictionary *userInfo = @{ NSLocalizedDescriptionKey : errorText };
    
    return [NSError errorWithDomain:@"HIDFramework"
                               code:code
                           userInfo:userInfo];
}

@end
