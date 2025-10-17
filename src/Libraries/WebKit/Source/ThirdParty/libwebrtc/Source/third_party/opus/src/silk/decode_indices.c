/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"

/* Decode side-information parameters from payload */
void silk_decode_indices(
    silk_decoder_state          *psDec,                         /* I/O  State                                       */
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int                    FrameIndex,                     /* I    Frame number                                */
    opus_int                    decode_LBRR,                    /* I    Flag indicating LBRR data is being decoded  */
    opus_int                    condCoding                      /* I    The type of conditional coding to use       */
)
{
    opus_int   i, k, Ix;
    opus_int   decode_absolute_lagIndex, delta_lagIndex;
    opus_int16 ec_ix[ MAX_LPC_ORDER ];
    opus_uint8 pred_Q8[ MAX_LPC_ORDER ];

    /*******************************************/
    /* Decode signal type and quantizer offset */
    /*******************************************/
    if( decode_LBRR || psDec->VAD_flags[ FrameIndex ] ) {
        Ix = ec_dec_icdf( psRangeDec, silk_type_offset_VAD_iCDF, 8 ) + 2;
    } else {
        Ix = ec_dec_icdf( psRangeDec, silk_type_offset_no_VAD_iCDF, 8 );
    }
    psDec->indices.signalType      = (opus_int8)silk_RSHIFT( Ix, 1 );
    psDec->indices.quantOffsetType = (opus_int8)( Ix & 1 );

    /****************/
    /* Decode gains */
    /****************/
    /* First subframe */
    if( condCoding == CODE_CONDITIONALLY ) {
        /* Conditional coding */
        psDec->indices.GainsIndices[ 0 ] = (opus_int8)ec_dec_icdf( psRangeDec, silk_delta_gain_iCDF, 8 );
    } else {
        /* Independent coding, in two stages: MSB bits followed by 3 LSBs */
        psDec->indices.GainsIndices[ 0 ]  = (opus_int8)silk_LSHIFT( ec_dec_icdf( psRangeDec, silk_gain_iCDF[ psDec->indices.signalType ], 8 ), 3 );
        psDec->indices.GainsIndices[ 0 ] += (opus_int8)ec_dec_icdf( psRangeDec, silk_uniform8_iCDF, 8 );
    }

    /* Remaining subframes */
    for( i = 1; i < psDec->nb_subfr; i++ ) {
        psDec->indices.GainsIndices[ i ] = (opus_int8)ec_dec_icdf( psRangeDec, silk_delta_gain_iCDF, 8 );
    }

    /**********************/
    /* Decode LSF Indices */
    /**********************/
    psDec->indices.NLSFIndices[ 0 ] = (opus_int8)ec_dec_icdf( psRangeDec, &psDec->psNLSF_CB->CB1_iCDF[ ( psDec->indices.signalType >> 1 ) * psDec->psNLSF_CB->nVectors ], 8 );
    silk_NLSF_unpack( ec_ix, pred_Q8, psDec->psNLSF_CB, psDec->indices.NLSFIndices[ 0 ] );
    celt_assert( psDec->psNLSF_CB->order == psDec->LPC_order );
    for( i = 0; i < psDec->psNLSF_CB->order; i++ ) {
        Ix = ec_dec_icdf( psRangeDec, &psDec->psNLSF_CB->ec_iCDF[ ec_ix[ i ] ], 8 );
        if( Ix == 0 ) {
            Ix -= ec_dec_icdf( psRangeDec, silk_NLSF_EXT_iCDF, 8 );
        } else if( Ix == 2 * NLSF_QUANT_MAX_AMPLITUDE ) {
            Ix += ec_dec_icdf( psRangeDec, silk_NLSF_EXT_iCDF, 8 );
        }
        psDec->indices.NLSFIndices[ i+1 ] = (opus_int8)( Ix - NLSF_QUANT_MAX_AMPLITUDE );
    }

    /* Decode LSF interpolation factor */
    if( psDec->nb_subfr == MAX_NB_SUBFR ) {
        psDec->indices.NLSFInterpCoef_Q2 = (opus_int8)ec_dec_icdf( psRangeDec, silk_NLSF_interpolation_factor_iCDF, 8 );
    } else {
        psDec->indices.NLSFInterpCoef_Q2 = 4;
    }

    if( psDec->indices.signalType == TYPE_VOICED )
    {
        /*********************/
        /* Decode pitch lags */
        /*********************/
        /* Get lag index */
        decode_absolute_lagIndex = 1;
        if( condCoding == CODE_CONDITIONALLY && psDec->ec_prevSignalType == TYPE_VOICED ) {
            /* Decode Delta index */
            delta_lagIndex = (opus_int16)ec_dec_icdf( psRangeDec, silk_pitch_delta_iCDF, 8 );
            if( delta_lagIndex > 0 ) {
                delta_lagIndex = delta_lagIndex - 9;
                psDec->indices.lagIndex = (opus_int16)( psDec->ec_prevLagIndex + delta_lagIndex );
                decode_absolute_lagIndex = 0;
            }
        }
        if( decode_absolute_lagIndex ) {
            /* Absolute decoding */
            psDec->indices.lagIndex  = (opus_int16)ec_dec_icdf( psRangeDec, silk_pitch_lag_iCDF, 8 ) * silk_RSHIFT( psDec->fs_kHz, 1 );
            psDec->indices.lagIndex += (opus_int16)ec_dec_icdf( psRangeDec, psDec->pitch_lag_low_bits_iCDF, 8 );
        }
        psDec->ec_prevLagIndex = psDec->indices.lagIndex;

        /* Get countour index */
        psDec->indices.contourIndex = (opus_int8)ec_dec_icdf( psRangeDec, psDec->pitch_contour_iCDF, 8 );

        /********************/
        /* Decode LTP gains */
        /********************/
        /* Decode PERIndex value */
        psDec->indices.PERIndex = (opus_int8)ec_dec_icdf( psRangeDec, silk_LTP_per_index_iCDF, 8 );

        for( k = 0; k < psDec->nb_subfr; k++ ) {
            psDec->indices.LTPIndex[ k ] = (opus_int8)ec_dec_icdf( psRangeDec, silk_LTP_gain_iCDF_ptrs[ psDec->indices.PERIndex ], 8 );
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        if( condCoding == CODE_INDEPENDENTLY ) {
            psDec->indices.LTP_scaleIndex = (opus_int8)ec_dec_icdf( psRangeDec, silk_LTPscale_iCDF, 8 );
        } else {
            psDec->indices.LTP_scaleIndex = 0;
        }
    }
    psDec->ec_prevSignalType = psDec->indices.signalType;

    /***************/
    /* Decode seed */
    /***************/
    psDec->indices.Seed = (opus_int8)ec_dec_icdf( psRangeDec, silk_uniform4_iCDF, 8 );
}
