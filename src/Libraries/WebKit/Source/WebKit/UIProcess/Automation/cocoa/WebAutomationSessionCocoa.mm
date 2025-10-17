/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#import "WebAutomationSession.h"

#if PLATFORM(COCOA)

#import "ViewSnapshotStore.h"
#import <wtf/FileSystem.h>

#if PLATFORM(IOS_FAMILY)
#import <ImageIO/CGImageDestination.h>
#import <MobileCoreServices/UTCoreTypes.h>
#import <WebCore/KeyEventCodesIOS.h>
#endif

namespace WebKit {
using namespace WebCore;

static std::optional<String> getBase64EncodedPNGData(const RetainPtr<CGImageRef>&& cgImage)
{
    RetainPtr<NSMutableData> imageData = adoptNS([[NSMutableData alloc] init]);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<CGImageDestinationRef> destination = adoptCF(CGImageDestinationCreateWithData((CFMutableDataRef)imageData.get(), kUTTypePNG, 1, 0));
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!destination)
        return std::nullopt;

    CGImageDestinationAddImage(destination.get(), cgImage.get(), 0);
    CGImageDestinationFinalize(destination.get());

    return String([imageData base64EncodedStringWithOptions:0]);
}


std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(ShareableBitmap::Handle&& imageDataHandle)
{
    auto bitmap = ShareableBitmap::create(WTFMove(imageDataHandle), SharedMemory::Protection::ReadOnly);
    if (!bitmap)
        return std::nullopt;

    return getBase64EncodedPNGData(bitmap->makeCGImage());
}

std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(const ViewSnapshot& snapshot)
{
    auto* snapshotSurface = snapshot.surface();
    if (!snapshotSurface)
        return std::nullopt;
    auto context = snapshotSurface->createPlatformContext();
    return getBase64EncodedPNGData(snapshotSurface->createImage(context.get()));
}

std::optional<String> WebAutomationSession::platformGenerateLocalFilePathForRemoteFile(const String& remoteFilePath, const String& base64EncodedFileContents)
{
    RetainPtr<NSData> fileContents = adoptNS([[NSData alloc] initWithBase64EncodedString:base64EncodedFileContents options:0]);
    if (!fileContents) {
        LOG_ERROR("WebAutomationSession: unable to decode base64-encoded file contents.");
        return std::nullopt;
    }

    NSString *temporaryDirectory = FileSystem::createTemporaryDirectory(@"WebDriver");
    NSURL *remoteFile = [NSURL fileURLWithPath:remoteFilePath isDirectory:NO];
    NSString *localFilePath = [temporaryDirectory stringByAppendingPathComponent:remoteFile.lastPathComponent];

    NSError *fileWriteError;
    [fileContents.get() writeToFile:localFilePath options:NSDataWritingAtomic error:&fileWriteError];
    if (fileWriteError) {
        LOG_ERROR("WebAutomationSession: Error writing image data to temporary file: %@", fileWriteError);
        return std::nullopt;
    }

    return String(localFilePath);
}

std::optional<unichar> WebAutomationSession::charCodeForVirtualKey(Inspector::Protocol::Automation::VirtualKey key) const
{
    switch (key) {
    case Inspector::Protocol::Automation::VirtualKey::Shift:
    case Inspector::Protocol::Automation::VirtualKey::ShiftRight:
    case Inspector::Protocol::Automation::VirtualKey::Control:
    case Inspector::Protocol::Automation::VirtualKey::ControlRight:
    case Inspector::Protocol::Automation::VirtualKey::Alternate:
    case Inspector::Protocol::Automation::VirtualKey::AlternateRight:
    case Inspector::Protocol::Automation::VirtualKey::Meta:
    case Inspector::Protocol::Automation::VirtualKey::MetaRight:
    case Inspector::Protocol::Automation::VirtualKey::Command:
    case Inspector::Protocol::Automation::VirtualKey::CommandRight:
        return std::nullopt;
    case Inspector::Protocol::Automation::VirtualKey::Help:
        return NSHelpFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Backspace:
        return NSBackspaceCharacter;
    case Inspector::Protocol::Automation::VirtualKey::Tab:
        return NSTabCharacter;
    case Inspector::Protocol::Automation::VirtualKey::Clear:
        return NSClearLineFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Enter:
        return NSEnterCharacter;
    case Inspector::Protocol::Automation::VirtualKey::Pause:
        return NSPauseFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Cancel:
        // The 'cancel' key does not exist on Apple keyboards and has no keycode.
        // According to the internet its functionality is similar to 'Escape'.
    case Inspector::Protocol::Automation::VirtualKey::Escape:
        return 0x1B;
    case Inspector::Protocol::Automation::VirtualKey::PageUp:
        return NSPageUpFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::PageDown:
        return NSPageDownFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::End:
        return NSEndFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Home:
        return NSHomeFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::LeftArrow:
        return NSLeftArrowFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::UpArrow:
        return NSUpArrowFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::RightArrow:
        return NSRightArrowFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::DownArrow:
        return NSDownArrowFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Insert:
        return NSInsertFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Delete:
        return NSDeleteFunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Space:
        return ' ';
    case Inspector::Protocol::Automation::VirtualKey::Semicolon:
        return ';';
    case Inspector::Protocol::Automation::VirtualKey::Equals:
        return '=';
    case Inspector::Protocol::Automation::VirtualKey::Return:
        return NSCarriageReturnCharacter;
    case Inspector::Protocol::Automation::VirtualKey::NumberPad0:
        return '0';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad1:
        return '1';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad2:
        return '2';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad3:
        return '3';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad4:
        return '4';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad5:
        return '5';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad6:
        return '6';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad7:
        return '7';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad8:
        return '8';
    case Inspector::Protocol::Automation::VirtualKey::NumberPad9:
        return '9';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadMultiply:
        return '*';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadAdd:
        return '+';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadSubtract:
        return '-';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadSeparator:
        // The 'Separator' key is only present on a few international keyboards.
        // It is usually mapped to the same character as Decimal ('.' or ',').
    case Inspector::Protocol::Automation::VirtualKey::NumberPadDecimal:
        return '.';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadDivide:
        return '/';
    case Inspector::Protocol::Automation::VirtualKey::Function1:
        return NSF1FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function2:
        return NSF2FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function3:
        return NSF3FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function4:
        return NSF4FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function5:
        return NSF5FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function6:
        return NSF6FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function7:
        return NSF7FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function8:
        return NSF8FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function9:
        return NSF9FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function10:
        return NSF10FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function11:
        return NSF11FunctionKey;
    case Inspector::Protocol::Automation::VirtualKey::Function12:
        return NSF12FunctionKey;
    default:
        return std::nullopt;
    }
}

std::optional<unichar> WebAutomationSession::charCodeIgnoringModifiersForVirtualKey(Inspector::Protocol::Automation::VirtualKey key) const
{
    switch (key) {
    case Inspector::Protocol::Automation::VirtualKey::NumberPadMultiply:
        return '8';
    case Inspector::Protocol::Automation::VirtualKey::NumberPadAdd:
        return '=';
    default:
        return charCodeForVirtualKey(key);
    }
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
