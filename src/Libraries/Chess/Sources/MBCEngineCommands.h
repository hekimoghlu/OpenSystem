/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#ifdef __cplusplus
#import "MBCBoard.h"

extern "C" {
#else
typedef unsigned MBCCompactMove;
#endif

extern MBCCompactMove MBCEncodeMove(const char * move, int ponder);
extern MBCCompactMove MBCEncodeDrop(const char * drop, int ponder);
extern MBCCompactMove MBCEncodeIllegal();
extern MBCCompactMove MBCEncodeLegal();
extern MBCCompactMove MBCEncodePong();
extern MBCCompactMove MBCEncodeStartGame();
extern MBCCompactMove MBCEncodeWhiteWins();
extern MBCCompactMove MBCEncodeBlackWins();
extern MBCCompactMove MBCEncodeDraw();
extern MBCCompactMove MBCEncodeTakeback();

extern void MBCIgnoredText(const char * text);
extern int MBCReadInput(char * buf, int max_size);

typedef void *          MBCLexerInstance;
extern void             MBCLexerInit(MBCLexerInstance*scanner);
extern void             MBCLexerDestroy(MBCLexerInstance scanner);
extern MBCCompactMove   MBCLexerScan(MBCLexerInstance scanner);

#ifdef __cplusplus
}
#endif
