/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#ifndef __NVCONTROL_H
#define __NVCONTROL_H

#define NV_CONTROL_ERRORS 0
#define NV_CONTROL_EVENTS 5
#define NV_CONTROL_NAME "NV-CONTROL"

#define NV_CONTROL_MAJOR 1
#define NV_CONTROL_MINOR 28

#define X_nvCtrlQueryExtension 0
#define X_nvCtrlIsNv 1
#define X_nvCtrlQueryAttribute 2
#define X_nvCtrlSetAttribute 3
#define X_nvCtrlQueryStringAttribute 4
#define X_nvCtrlQueryValidAttributeValues 5
#define X_nvCtrlSelectNotify 6
#define X_nvCtrlSetGvoColorConversionDeprecated 7
#define X_nvCtrlQueryGvoColorConversionDeprecated 8
#define X_nvCtrlSetStringAttribute 9
/* STUB X_nvCtrlQueryDDCCILutSize                   10 */
/* STUB X_nvCtrlQueryDDCCISinglePointLutOperation   11 */
/* STUB X_nvCtrlSetDDCCISinglePointLutOperation     12 */
/* STUB X_nvCtrlQueryDDCCIBlockLutOperation         13 */
/* STUB X_nvCtrlSetDDCCIBlockLutOperation           14 */
/* STUB X_nvCtrlSetDDCCIRemoteProcedureCall         15 */
/* STUB X_nvCtrlQueryDDCCIDisplayControllerType     16 */
/* STUB X_nvCtrlQueryDDCCICapabilities              17 */
/* STUB X_nvCtrlQueryDDCCITimingReport              18 */
#define X_nvCtrlSetAttributeAndGetStatus 19
#define X_nvCtrlQueryBinaryData 20
#define X_nvCtrlSetGvoColorConversion 21
#define X_nvCtrlQueryGvoColorConversion 22
#define X_nvCtrlSelectTargetNotify 23
#define X_nvCtrlQueryTargetCount 24
#define X_nvCtrlStringOperation 25
#define X_nvCtrlQueryValidAttributeValues64 26
#define X_nvCtrlQueryAttribute64 27
#define X_nvCtrlQueryValidStringAttributeValues 28
#define X_nvCtrlQueryAttributePermissions 29
#define X_nvCtrlQueryStringAttributePermissions 30
#define X_nvCtrlQueryBinaryDataAttributePermissions 31
#define X_nvCtrlQueryStringOperationAttributePermissions 32

#define X_nvCtrlLastRequest (X_nvCtrlQueryStringOperationAttributePermissions + 1)

/* Define 32 bit floats */
typedef float FLOAT32;
#ifndef F32
#    define F32
#endif

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
} xnvCtrlQueryExtensionReq;
#define sz_xnvCtrlQueryExtensionReq 4

typedef struct
{
    BYTE type; /* X_Reply */
    CARD8 padb1;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD16 major B16;
    CARD16 minor B16;
    CARD32 padl4 B32;
    CARD32 padl5 B32;
    CARD32 padl6 B32;
    CARD32 padl7 B32;
    CARD32 padl8 B32;
} xnvCtrlQueryExtensionReply;
#define sz_xnvCtrlQueryExtensionReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;
} xnvCtrlIsNvReq;
#define sz_xnvCtrlIsNvReq 8

typedef struct
{
    BYTE type; /* X_Reply */
    CARD8 padb1;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 isnv B32;
    CARD32 padl4 B32;
    CARD32 padl5 B32;
    CARD32 padl6 B32;
    CARD32 padl7 B32;
    CARD32 padl8 B32;
} xnvCtrlIsNvReply;
#define sz_xnvCtrlIsNvReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 target_type B32;
} xnvCtrlQueryTargetCountReq;
#define sz_xnvCtrlQueryTargetCountReq 8

typedef struct
{
    BYTE type; /* X_Reply */
    CARD8 padb1;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 count B32;
    CARD32 padl4 B32;
    CARD32 padl5 B32;
    CARD32 padl6 B32;
    CARD32 padl7 B32;
    CARD32 padl8 B32;
} xnvCtrlQueryTargetCountReply;
#define sz_xnvCtrlQueryTargetCountReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;   /* X screen number or GPU number */
    CARD16 target_type B16; /* X screen or GPU */
    CARD32 display_mask B32;
    CARD32 attribute B32;
} xnvCtrlQueryAttributeReq;
#define sz_xnvCtrlQueryAttributeReq 16

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    INT32 value B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlQueryAttributeReply;
#define sz_xnvCtrlQueryAttributeReply 32

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    CARD32 pad3 B32;
    int64_t value_64;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlQueryAttribute64Reply;
#define sz_xnvCtrlQueryAttribute64Reply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;
    CARD16 target_type B16;
    CARD32 display_mask B32;
    CARD32 attribute B32;
    INT32 value B32;
} xnvCtrlSetAttributeReq;
#define sz_xnvCtrlSetAttributeReq 20

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;
    CARD16 target_type B16;
    CARD32 display_mask B32;
    CARD32 attribute B32;
    INT32 value B32;
} xnvCtrlSetAttributeAndGetStatusReq;
#define sz_xnvCtrlSetAttributeAndGetStatusReq 20

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    CARD32 pad3 B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlSetAttributeAndGetStatusReply;
#define sz_xnvCtrlSetAttributeAndGetStatusReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;   /* X screen number or GPU number */
    CARD16 target_type B16; /* X screen or GPU */
    CARD32 display_mask B32;
    CARD32 attribute B32;
} xnvCtrlQueryStringAttributeReq;
#define sz_xnvCtrlQueryStringAttributeReq 16

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    CARD32 n B32; /* Length of string */
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlQueryStringAttributeReply;
#define sz_xnvCtrlQueryStringAttributeReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;
    CARD16 target_type B16;
    CARD32 display_mask B32;
    CARD32 attribute B32;
    CARD32 num_bytes B32;
} xnvCtrlSetStringAttributeReq;
#define sz_xnvCtrlSetStringAttributeReq 20

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    CARD32 pad3 B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlSetStringAttributeReply;
#define sz_xnvCtrlSetStringAttributeReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;   /* X screen number or GPU number */
    CARD16 target_type B16; /* X screen or GPU */
    CARD32 display_mask B32;
    CARD32 attribute B32;
} xnvCtrlQueryValidAttributeValuesReq;
#define sz_xnvCtrlQueryValidAttributeValuesReq 16

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    INT32 attr_type B32;
    INT32 min B32;
    INT32 max B32;
    CARD32 bits B32;
    CARD32 perms B32;
} xnvCtrlQueryValidAttributeValuesReply;
#define sz_xnvCtrlQueryValidAttributeValuesReply 32

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    INT32 attr_type B32;
    int64_t min_64;
    int64_t max_64;
    CARD64 bits_64;
    CARD32 perms B32;
    CARD32 pad1 B32;
} xnvCtrlQueryValidAttributeValues64Reply;
#define sz_xnvCtrlQueryValidAttributeValues64Reply 48
#define sz_xnvCtrlQueryValidAttributeValues64Reply_extra ((48 - 32) >> 2)

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 attribute B32;
} xnvCtrlQueryAttributePermissionsReq;
#define sz_xnvCtrlQueryAttributePermissionsReq 8

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    INT32 attr_type B32;
    CARD32 perms B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
    CARD32 pad8 B32;
} xnvCtrlQueryAttributePermissionsReply;
#define sz_xnvCtrlQueryAttributePermissionsReply 32

/* Set GVO Color Conversion request (deprecated) */
typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;
    FLOAT32 row1_col1 F32;
    FLOAT32 row1_col2 F32;
    FLOAT32 row1_col3 F32;
    FLOAT32 row1_col4 F32;
    FLOAT32 row2_col1 F32;
    FLOAT32 row2_col2 F32;
    FLOAT32 row2_col3 F32;
    FLOAT32 row2_col4 F32;
    FLOAT32 row3_col1 F32;
    FLOAT32 row3_col2 F32;
    FLOAT32 row3_col3 F32;
    FLOAT32 row3_col4 F32;
} xnvCtrlSetGvoColorConversionDeprecatedReq;
#define sz_xnvCtrlSetGvoColorConversionDeprecatedReq 56

/* Query GVO Color Conversion request (deprecated) */
typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;
} xnvCtrlQueryGvoColorConversionDeprecatedReq;
#define sz_xnvCtrlQueryGvoColorConversionDeprecatedReq 8

/* Query GVO Color Conversion reply (deprecated) */
typedef struct
{
    BYTE type; /* X_Reply */
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 pad3 B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
    CARD32 pad8 B32;
} xnvCtrlQueryGvoColorConversionDeprecatedReply;
#define sz_xnvCtrlQueryGvoColorConversionDeprecatedReply 32

/* Set GVO Color Conversion request */
typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;

    FLOAT32 cscMatrix_y_r F32;
    FLOAT32 cscMatrix_y_g F32;
    FLOAT32 cscMatrix_y_b F32;

    FLOAT32 cscMatrix_cr_r F32;
    FLOAT32 cscMatrix_cr_g F32;
    FLOAT32 cscMatrix_cr_b F32;

    FLOAT32 cscMatrix_cb_r F32;
    FLOAT32 cscMatrix_cb_g F32;
    FLOAT32 cscMatrix_cb_b F32;

    FLOAT32 cscOffset_y F32;
    FLOAT32 cscOffset_cr F32;
    FLOAT32 cscOffset_cb F32;

    FLOAT32 cscScale_y F32;
    FLOAT32 cscScale_cr F32;
    FLOAT32 cscScale_cb F32;

} xnvCtrlSetGvoColorConversionReq;
#define sz_xnvCtrlSetGvoColorConversionReq 68

/* Query GVO Color Conversion request */
typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;
} xnvCtrlQueryGvoColorConversionReq;
#define sz_xnvCtrlQueryGvoColorConversionReq 8

/* Query GVO Color Conversion reply */
typedef struct
{
    BYTE type; /* X_Reply */
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 pad3 B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
    CARD32 pad8 B32;
} xnvCtrlQueryGvoColorConversionReply;
#define sz_xnvCtrlQueryGvoColorConversionReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;   /* X screen number or GPU number */
    CARD16 target_type B16; /* X screen or GPU */
    CARD32 display_mask B32;
    CARD32 attribute B32;
} xnvCtrlQueryBinaryDataReq;
#define sz_xnvCtrlQueryBinaryDataReq 16

typedef struct
{
    BYTE type;
    BYTE pad0;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 flags B32;
    CARD32 n B32;
    CARD32 pad4 B32;
    CARD32 pad5 B32;
    CARD32 pad6 B32;
    CARD32 pad7 B32;
} xnvCtrlQueryBinaryDataReply;
#define sz_xnvCtrlQueryBinaryDataReply 32

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD32 screen B32;
    CARD16 notifyType B16;
    CARD16 onoff B16;
} xnvCtrlSelectNotifyReq;
#define sz_xnvCtrlSelectNotifyReq 12

typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_id B16;   /* X screen number or GPU number */
    CARD16 target_type B16; /* X screen or GPU */
    CARD32 display_mask B32;
    CARD32 attribute B32;
    CARD32 num_bytes B32; /* Length of string */
} xnvCtrlStringOperationReq;
#define sz_xnvCtrlStringOperationReq 20

typedef struct
{
    BYTE type; /* X_Reply */
    CARD8 padb1;
    CARD16 sequenceNumber B16;
    CARD32 length B32;
    CARD32 ret B32;
    CARD32 num_bytes B32; /* Length of string */
    CARD32 padl4 B32;
    CARD32 padl5 B32;
    CARD32 padl6 B32;
    CARD32 padl7 B32;
} xnvCtrlStringOperationReply;
#define sz_xnvCtrlStringOperationReply 32

typedef struct
{
    union
    {
        struct
        {
            BYTE type;
            BYTE detail;
            CARD16 sequenceNumber B16;
        } u;
        struct
        {
            BYTE type;
            BYTE detail;
            CARD16 sequenceNumber B16;
            CARD32 time B32;
            CARD32 screen B32;
            CARD32 display_mask B32;
            CARD32 attribute B32;
            CARD32 value B32;
            CARD32 pad0 B32;
            CARD32 pad1 B32;
        } attribute_changed;
    } u;
} xnvctrlEvent;

/*
 * Leave target_type before target_id for the
 * xnvCtrlSelectTargetNotifyReq and xnvctrlEventTarget
 * structures, even though other request protocol structures
 * store target_id in the bottom 16-bits of the second DWORD of the
 * structures.  The event-related structures were added in version
 * 1.8, and so there is no prior version with which to maintain
 * compatibility.
 */
typedef struct
{
    CARD8 reqType;
    CARD8 nvReqType;
    CARD16 length B16;
    CARD16 target_type B16; /* Don't swap these */
    CARD16 target_id B16;
    CARD16 notifyType B16;
    CARD16 onoff B16;
} xnvCtrlSelectTargetNotifyReq;
#define sz_xnvCtrlSelectTargetNotifyReq 12

typedef struct
{
    union
    {
        struct
        {
            BYTE type;
            BYTE detail;
            CARD16 sequenceNumber B16;
        } u;
        struct
        {
            BYTE type;
            BYTE detail;
            CARD16 sequenceNumber B16;
            CARD32 time B32;
            CARD16 target_type B16; /* Don't swap these */
            CARD16 target_id B16;
            CARD32 display_mask B32;
            CARD32 attribute B32;
            CARD32 value B32;
            CARD32 pad0 B32;
            CARD32 pad1 B32;
        } attribute_changed;
        struct
        {
            BYTE type;
            BYTE detail;
            CARD16 sequenceNumber B16;
            CARD32 time B32;
            CARD16 target_type B16; /* Don't swap these */
            CARD16 target_id B16;
            CARD32 display_mask B32;
            CARD32 attribute B32;
            CARD32 value B32;
            CARD8 availability;
            CARD8 pad0;
            CARD16 pad1 B16;
            CARD32 pad2 B32;
        } availability_changed;
    } u;
} xnvctrlEventTarget;

#endif /* __NVCONTROL_H */
