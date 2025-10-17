/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

// Copyright (C) 2024 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#if os(visionOS)

import LinearMediaKit
import WebKitSwift

// MARK: Objective-C Implementations

@_objcImplementation extension WKSLinearMediaContentMetadata {
    let title: String?
    let subtitle: String?
    
    init(title: String?, subtitle: String?) {
        self.title = title
        self.subtitle = subtitle
    }
}

@_objcImplementation extension WKSLinearMediaTimeRange {
    let lowerBound: TimeInterval
    let upperBound: TimeInterval

    init(lowerBound: TimeInterval, upperBound: TimeInterval) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }
}

@_objcImplementation extension WKSLinearMediaTrack {
    let localizedDisplayName: String

    init(localizedDisplayName: String) {
        self.localizedDisplayName = localizedDisplayName
    }
}

@_objcImplementation extension WKSLinearMediaSpatialVideoMetadata {
    let width: Int32
    let height: Int32
    let horizontalFOVDegrees: Float
    let baseline: Float
    let disparityAdjustment: Float

    init(width: Int32, height: Int32, horizontalFOVDegrees: Float, baseline: Float, disparityAdjustment: Float) {
        self.width = width
        self.height = height
        self.horizontalFOVDegrees = horizontalFOVDegrees
        self.baseline = baseline
        self.disparityAdjustment = disparityAdjustment
    }
}

// MARK: LinearMediaKit Extensions

extension WKSLinearMediaContentMetadata {
    var contentMetadata: ContentMetadataContainer {
        var container = ContentMetadataContainer()
        container.displayTitle = title
        container.displaySubtitle = subtitle
        return container
    }
}

extension WKSLinearMediaContentMode {
    init(_ contentMode: ContentMode?) {
        switch contentMode {
        case .scaleAspectFit?:
            self = .scaleAspectFit
        case .scaleAspectFill?:
            self = .scaleAspectFill
        case .scaleToFill?:
            self = .scaleToFill
        case .none:
            self = .none
        @unknown default:
            fatalError()
        }
    }

    var contentMode: ContentMode? {
        switch self {
        case .none:
            nil
        case .scaleAspectFit:
            .scaleAspectFit
        case .scaleAspectFill:
            .scaleAspectFill
        case .scaleToFill:
            .scaleToFill
        @unknown default:
            fatalError()
        }
    }

    static var `default`: WKSLinearMediaContentMode {
        .init(.default)
    }
}

extension WKSLinearMediaContentType {
    var contentType: ContentType? {
        switch self {
        case .none:
            nil
        case .immersive:
            .immersive
        case .spatial:
            .spatial
        case .planar:
            .planar
        case .audioOnly:
            .audioOnly
        @unknown default:
            fatalError()
        }
    }
}

extension WKSLinearMediaPresentationState: CustomStringConvertible {
    public var description: String {
        switch self {
        case .inline:
            return "inline"
        case .enteringFullscreen:
            return "enteringFullscreen"
        case .fullscreen:
            return "fullscreen"
        case .exitingFullscreen:
            return "exitingFullscreen"
        @unknown default:
            fatalError()
        }
    }
}

extension WKSLinearMediaViewingMode: CustomStringConvertible {
    init(_ viewingMode: ViewingMode?) {
        switch viewingMode {
        case .mono?:
            self = .mono
        case .stereo?:
            self = .stereo
        case .immersive?:
            self = .immersive
        case .spatial?:
            self = .spatial
        case .none:
            self = .none
        @unknown default:
            fatalError()
        }
    }
    
    var viewingMode: ViewingMode? {
        switch self {
        case .none:
            nil
        case .mono:
            .mono
        case .stereo:
            .stereo
        case .immersive:
            .immersive
        case .spatial:
            .spatial
        @unknown default:
            fatalError()
        }
    }

    public var description: String {
        switch self {
        case .none:
            return "none"
        case .mono:
            return "mono"
        case .stereo:
            return "stereo"
        case .immersive:
            return "immersive"
        case .spatial:
            return "spatial"
        @unknown default:
            fatalError()
        }
    }
}

extension WKSLinearMediaFullscreenBehaviors {
    init(_ fullscreenBehaviors: FullscreenBehaviors) {
        self = .init(rawValue: fullscreenBehaviors.rawValue)
    }

    var fullscreenBehaviors: FullscreenBehaviors {
        .init(rawValue: self.rawValue)
    }
}

extension WKSLinearMediaTimeRange {
    var closedRange: ClosedRange<TimeInterval> {
        return lowerBound...upperBound
    }

    var range: Range<TimeInterval> {
        return lowerBound..<upperBound
    }
}

extension WKSLinearMediaTrack: @retroactive Track {
}

extension WKSLinearMediaSpatialVideoMetadata {
    var metadata: SpatialVideoMetadata {
        return SpatialVideoMetadata(width: self.width, height: self.height, horizontalFOVDegrees: self.horizontalFOVDegrees, baseline: self.baseline, disparityAdjustment: self.disparityAdjustment, isRecommendedForImmersive: true)
    }
}

#endif // os(visionOS)
