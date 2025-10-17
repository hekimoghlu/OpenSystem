/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
/*
**
**  NAME
**
**      cnfbuf.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Fragment buffer management routines for connection based
**  protocol services.
**
**
*/

#include <commonp.h>    /* Common declarations for all RPC runtime */
#include <com.h>        /* More common declarations */
#include <cnp.h>        /* Connection common declarations */
#include <cnfbuf.h>	/* Fragment buffer declarations */

GLOBAL unsigned32 rpc_g_cn_large_frag_size = RPC_C_CN_LARGE_FRAG_SIZE;


/*
**++
**
**  ROUTINE NAME:       rpc__cn_fragbuf_free
**
**  SCOPE:              PRIVATE
**
**  DESCRIPTION:
**
**  Deallocates a large fragment buffer.
**
**  INPUTS:
**
**      buffer_p        Pointer to the large fragment buffer which is to be
**                      deallocated.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    lg_fragbuf_list
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE void rpc__cn_fragbuf_free
(
   rpc_cn_fragbuf_p_t buffer_p
)
{
#ifdef MAX_DEBUG
    memset ((char *) buffer_p->data_area, 0, rpc_g_cn_large_frag_size);
    memset ((char *) buffer_p, 0, sizeof (rpc_cn_fragbuf_t));
#endif
    rpc__list_element_free (&rpc_g_cn_lg_fbuf_lookaside_list,
                            (dce_pointer_t) buffer_p);
}

/*
**++
**
**  ROUTINE NAME:       rpc__cn_smfragbuf_free
**
**  SCOPE:              PRIVATE
**
**  DESCRIPTION:
**
**  Deallocates a small fragment buffer.
**
**  INPUTS:
**
**      buffer_p        Pointer to the small fragment buffer which is to be
**                      deallocated.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    sm_fragbuf_list
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE void rpc__cn_smfragbuf_free
(
   rpc_cn_fragbuf_p_t      buffer_p
)
{
#ifdef MAX_DEBUG
    memset ((char *) buffer_p->data_area, 0, RPC_C_CN_SMALL_FRAG_SIZE);
    memset ((char *) buffer_p, 0, sizeof (rpc_cn_fragbuf_t));
#endif
    rpc__list_element_free (&rpc_g_cn_sm_fbuf_lookaside_list,
                            (dce_pointer_t) buffer_p );
}

/*
**++
**
**  ROUTINE NAME:       rpc__cn_fragbuf_alloc
**
**  SCOPE:              PRIVATE
**
**  DESCRIPTION:
**
**  Allocates a fragment buffer and returns a pointer to it.
**
**  INPUTS:
**
**      alloc_large_buf If TRUE, then allocates a large fragment
**                      buffer.  Otherwise, allocates a small one.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     Address of the allocated fragment buffer.
**
**  SIDE EFFECTS:       none
**
**--
**/

PRIVATE rpc_cn_fragbuf_p_t rpc__cn_fragbuf_alloc
(
    boolean32               alloc_large_buf
)
{
    rpc_cn_fragbuf_p_t  fbp;

    /*
     * Get a fragment buffer from the appropriate lookaside list
     */

    if (alloc_large_buf)
    {
	fbp = (rpc_cn_fragbuf_p_t)
            rpc__list_element_alloc (&rpc_g_cn_lg_fbuf_lookaside_list,
                                     true);
        if (fbp != NULL)
        {
            fbp->fragbuf_dealloc = rpc__cn_fragbuf_free;
            fbp->max_data_size = rpc_g_cn_large_frag_size;
        }
        else
        {
            return (NULL);
        }
    }
    else
    {
        fbp = (rpc_cn_fragbuf_p_t)
            rpc__list_element_alloc (&rpc_g_cn_sm_fbuf_lookaside_list,
                                     true);
        if (fbp != NULL)
        {
            fbp->fragbuf_dealloc = rpc__cn_smfragbuf_free;
            fbp->max_data_size = RPC_C_CN_SMALL_FRAG_SIZE;
        }
        else
        {
            return (NULL);
        }
    }

    /*
     * Set the data pointer to an 8 byte aligned boundary.
     */

    fbp->data_p = (dce_pointer_t) RPC_CN_ALIGN_PTR(fbp->data_area, 8);
    memset (fbp->data_area, 0, fbp->max_data_size);

    /*
     * Set up the size of the data being pointed to.
     */
    fbp->data_size = 0;

    /*
     * Return a pointer to the "filled-in" fragment buffer
     */
    return (fbp);
}

/*
**++
**
**  ROUTINE NAME:       rpc__cn_dynfragbuf_free
**
**  SCOPE:              INTERNAL
**
**  DESCRIPTION:
**
**  Deallocates a dynamic fragment buffer.
**
**  INPUTS:
**
**      buffer_p        Pointer to the dynamic fragment buffer which is to be
**                      deallocated.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/

INTERNAL void rpc__cn_dynfragbuf_free
(
   rpc_cn_fragbuf_p_t buffer_p
)
{
#ifdef MAX_DEBUG
    memset ((char *) buffer_p->data_area, 0, buffer_p->max_data_size);
    memset ((char *) buffer_p, 0, sizeof (rpc_cn_fragbuf_t));
#endif

    RPC_MEM_FREE(buffer_p, RPC_C_MEM_CN_PAC_BUF);
}

/*
**++
**
**  ROUTINE NAME:       rpc__cn_fragbuf_alloc_dyn
**
**  SCOPE:              PRIVATE
**
**  DESCRIPTION:
**
**  Allocates a fragment buffer and returns a pointer to it.
**
**  INPUTS:
**
**      alloc_size The size of the dynamic allocated fragment buffer.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            none
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     Address of the allocated fragment buffer.
**
**  SIDE EFFECTS:       none
**
**--
**/
PRIVATE rpc_cn_fragbuf_p_t rpc__cn_fragbuf_alloc_dyn
(
    unsigned32               alloc_size
)
{
    rpc_cn_fragbuf_p_t  fbp;

    RPC_MEM_ALLOC (fbp,
                   rpc_cn_fragbuf_p_t,
                   sizeof(rpc_cn_fragbuf_t) + alloc_size,
                   RPC_C_MEM_CN_PAC_BUF,
                   RPC_C_MEM_NOWAIT);

    if (fbp != NULL)
    {
        memset(fbp, 0, sizeof(rpc_cn_fragbuf_t));
        fbp->fragbuf_dealloc = rpc__cn_dynfragbuf_free;
        fbp->max_data_size = alloc_size;
    }
    else
    {
        return (NULL);
    }

    /*
     * Set the data pointer to an 8 byte aligned boundary.
     */

    fbp->data_p = (dce_pointer_t) RPC_CN_ALIGN_PTR(fbp->data_area, 8);
    memset(fbp->data_area, 0, fbp->max_data_size);

    /*
     * Set up the size of the data being pointed to.
     */
    fbp->data_size = 0;

    /*
     * Return a pointer to the "filled-in" fragment buffer
     */
    return (fbp);
}
