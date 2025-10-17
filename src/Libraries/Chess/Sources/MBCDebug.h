/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
class MBCDebug {
public:
    static void Update();
    static bool ShowDebugMenu()                 { return DebugFlags() & 1;  }
    static bool LogMouse()                      { return DebugFlags() & 2;  }
    static void SetLogMouse(bool val)           { SetDebugFlags(2, val);    }
    static bool LogStart()                      { return DebugFlags() & 4;  }
    static bool DumpLanguageModels()            { return DebugFlags() & 8;  }
    static void SetDumpLanguageModels(bool val) { SetDebugFlags(8, val);    }
    static bool Use1xTextures()                 { return DebugFlags() & 16; }
private:
    static int  DebugFlags()                    { return sDebugFlags;       }
    static void SetDebugFlags(int flag, bool val);
    static int  sDebugFlags;
};
