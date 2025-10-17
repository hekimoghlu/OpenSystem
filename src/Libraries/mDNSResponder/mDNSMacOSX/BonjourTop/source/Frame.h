/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
//  Frame.h
//  TestTB
//
//  Created by Terrin Eager on 1/19/13.
//
//

#ifndef __TestTB__Frame__
#define __TestTB__Frame__

#include "bjtypes.h"
#include "bjIPAddr.h"
#include "bjMACAddr.h"

class Frame
{
public:
    void Set(BJ_UINT8* data,BJ_UINT32 len,BJ_UINT64 t);
    BJ_UINT8* GetEthernetStart();
    BJ_UINT8* GetIPStart();
    BJ_UINT8* GetUDPStart();
    BJ_UINT8* GetBonjourStart();

    BJIPAddr* GetSrcIPAddr();
    BJIPAddr* GetDestIPAddr();

    BJMACAddr* GetSrcMACAddr();
    BJMACAddr* GetDestMACAddr();

    int m_bCurrentFrameIPversion;

    BJ_UINT64 GetTime(){ return frameTime; };

    enum BJ_DATALINKTYPE {
        BJ_DLT_EN10MB = 1,
        BJ_DLT_IEEE802_11=105
    };

    void SetDatalinkType (BJ_DATALINKTYPE datalinkType);
private:

    BJ_UINT32 GetLinklayerHeaderLength();

    //Get the header length of the current 802.11 frame.
    BJ_UINT32 Get80211HeaderLength();

    BJ_UINT8* frameData;
    BJ_UINT32 length;

    BJIPAddr sourceIPAddr;
    BJIPAddr destIPAddr;

    BJMACAddr sourceMACAddr;
    BJMACAddr destMACAddr;

    BJ_UINT64 frameTime; // in microseconds


    BJ_DATALINKTYPE m_datalinkType = BJ_DLT_EN10MB;


};


#endif /* defined(__TestTB__Frame__) */
