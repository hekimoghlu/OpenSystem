/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
//  CaptureFile.h
//  TestTB
//
//  Created by Terrin Eager on 9/14/12.
//
//

#ifndef __TestTB__CaptureFile__
#define __TestTB__CaptureFile__

#include <iostream>
#include <pcap/pcap.h>
#include "bjtypes.h"
#include "bjsocket.h"
#include "Frame.h"

class CCaptureFile
{
public:
    CCaptureFile();
    virtual ~CCaptureFile();
    bool Open(const char* pFileName);
    bool NextFrame();
    bool Close();

    Frame m_CurrentFrame;



    time_t GetDeltaTime();

    __uint32_t GetBufferLen(BJ_UINT8* pStart);

    __uint32_t GetWiredLength(){ return m_nWireLen;};


private:
    bool Init();
    bool Clear();

    pcap_t* m_hPCap;
    BJ_UINT8* m_pFrameData;
    __uint32_t  m_nCaptureLen;
    __uint32_t  m_nWireLen;
    time_t m_TimeSec;

    time_t m_nFirstFrameTime;

    Frame::BJ_DATALINKTYPE m_datalinkType;
    bool m_bFormatIsPCapNG;
};

#endif /* defined(__TestTB__CaptureFile__) */
