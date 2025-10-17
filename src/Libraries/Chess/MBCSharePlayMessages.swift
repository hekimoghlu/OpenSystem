/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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

//
//  MBCSharePlayMessages.swift
//  MBCSharePlayMessages
//

import Foundation
import os.log

@objc(StartSelectionMessage)
public class StartSelectionMessage: NSObject, Codable {
    @objc public var square: CUnsignedChar
}

@objc(EndSelectionMessage)
public class EndSelectionMessage: NSObject, Codable {
    @objc public var square: CUnsignedChar
    @objc public var animate: Bool
}

@objc(SharePlaySettingsMessage)
public class SharePlaySettingsMessage: NSObject, Codable {
    @objc public var isPlayer: Bool
    @objc public var disconnecting: Bool
}

@objc(SharePlayBoardStateMessage)
public class SharePlayBoardStateMessage: NSObject, Codable {
    @objc public var fen: String
    @objc public var holding: String
    @objc public var moves: String
    @objc public var numMoves: Int32

    @objc public init(fen: String, holding: String, moves:String, numMoves:Int32) {
        self.fen = fen
        self.holding = holding
        self.moves = moves
        self.numMoves = numMoves
    }
}

enum MessageType: Int, Codable {
    case takeBack
}

struct GenericMessage: Codable {
    let type: MessageType
}

struct BoardMessage: Codable {
    let fen: String
    let holding: String
    let moves: String
    let numMoves: Int32
}
