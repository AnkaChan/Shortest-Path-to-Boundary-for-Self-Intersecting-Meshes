#pragma once
#include "cuda_runtime.h"
#include "../MatrixOps/CuMatrix.h"
#include "../MatrixOps/CuMatrixDefs.h"
#include "device_launch_parameters.h"

namespace CuMatrix
{
    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC DType tetOrientedVolume(DType* allVertsArray, int32_t* tetVIds) {
        float* tvs[4] = {
            allVertsArray + PointVecDims * tetVIds[0],
            allVertsArray + PointVecDims * tetVIds[1],
            allVertsArray + PointVecDims * tetVIds[2],
            allVertsArray + PointVecDims * tetVIds[3],
        };

        DType AB[3];
        vec3Minus(tvs[1], tvs[0], AB);
        DType AC[3];
        vec3Minus(tvs[2], tvs[0], AC);
        DType AD[3];
        vec3Minus(tvs[3], tvs[0], AD);

        DType tetOrientedVol = vec3TripleProduct(AB, AC, AD);

        return tetOrientedVol;
    }

    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC DType tetOrientedVolume(DType* v1, DType* v2, DType* v3, DType* v4) {

        DType AB[3];
        vec3Minus(v2, v1, AB);
        DType AC[3];
        vec3Minus(v3, v1, AC);
        DType AD[3];
        vec3Minus(v4, v1, AD);

        DType tetOrientedVol = vec3TripleProduct(AB, AC, AD);

        return tetOrientedVol;
    }

    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC void  tetCentroid(DType* p, DType* allVertsArray, int32_t* tetVIds) {
        vec3Set(p, DType(0.f));

        float* tvs[4] = {
            allVertsArray + PointVecDims * tetVIds[0],
            allVertsArray + PointVecDims * tetVIds[1],
            allVertsArray + PointVecDims * tetVIds[2],
            allVertsArray + PointVecDims * tetVIds[3],
        };
        vec3Add(p, tvs[0], p);
        vec3Add(p, tvs[1], p);
        vec3Add(p, tvs[2], p);
        vec3Add(p, tvs[3], p);

        vec3Mul(p, 0.25f, p);

    }

    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC bool tetPointInTet(DType* p, DType* allVertsArray, int32_t* tetVIds) {
        float* tvs[4] = {
            allVertsArray + PointVecDims * tetVIds[0],
            allVertsArray + PointVecDims * tetVIds[1],
            allVertsArray + PointVecDims * tetVIds[2],
            allVertsArray + PointVecDims * tetVIds[3],
        };

        DType AB[3];
        vec3Minus(tvs[1], tvs[0], AB);
        DType AC[3];
        vec3Minus(tvs[2], tvs[0], AC);
        DType AD[3];
        vec3Minus(tvs[3], tvs[0], AD);

        DType tetOrientedVol = vec3TripleProduct(AB, AC, AD);

        const int32_t order[4][3] = { { 1, 2, 3 },{ 2, 0, 3 },{ 0, 1, 3 },{ 1, 0, 2 } };

        for (int32_t i = 0; i < 4; ++i) {

            DType v1[3]; // = vs4[order[i][1]] - vs4[order[i][0]]; // HalfEdgeVec(pHE1);
            vec3Minus(tvs[order[i][1]], tvs[order[i][0]], v1);

            DType v2[3]; // = vs4[order[i][2]] - vs4[order[i][1]];  // HalfEdgeVec(pHE2);
            vec3Minus(tvs[order[i][2]], tvs[order[i][1]], v2);

            DType vp[3];
            vec3Minus(p, tvs[order[i][0]], vp);

            if (vec3TripleProduct(vp, v1, v2) * tetOrientedVol >= 0)
            {
                return false;
            }
        }

        return true;
    }

    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC bool tetPointBarycentricsInTet(DType* p, DType* allVertsArray, int32_t* tetVIds, DType* barycentrics) {
        float* tvs[4] = {
            allVertsArray + PointVecDims * tetVIds[0],
            allVertsArray + PointVecDims * tetVIds[1],
            allVertsArray + PointVecDims * tetVIds[2],
            allVertsArray + PointVecDims * tetVIds[3],
        };

        DType AB[3];
        vec3Minus(tvs[1], tvs[0], AB);
        DType AC[3];
        vec3Minus(tvs[2], tvs[0], AC);
        DType AD[3];
        vec3Minus(tvs[3], tvs[0], AD);

        DType tetOrientedVol = vec3TripleProduct(AB, AC, AD);

        const int32_t order[4][3] = { { 1, 2, 3 },{ 2, 0, 3 },{ 0, 1, 3 },{ 1, 0, 2 } };

        for (int32_t i = 0; i < 3; ++i) {

            DType v1[3]; // = vs4[order[i][1]] - vs4[order[i][0]]; // HalfEdgeVec(pHE1);
            vec3Minus(tvs[order[i][1]], tvs[order[i][0]], v1);

            DType v2[3]; // = vs4[order[i][2]] - vs4[order[i][1]];  // HalfEdgeVec(pHE2);
            vec3Minus(tvs[order[i][2]], tvs[order[i][1]], v2);

            DType vp[3];
            vec3Minus(p, tvs[order[i][0]], vp);

            DType subTetOrientedVol = (vec3TripleProduct(vp, v1, v2));

            barycentrics[i] = - subTetOrientedVol / tetOrientedVol;
        }
        barycentrics[3] = 1.f - barycentrics[0] - barycentrics[1] - barycentrics[2];
        return true;
    }

    template<typename DType, int PointVecDims = 3>
    GPU_CPU_INLINE_FUNC void  triangleOrientedArea(DType* allVertsArray, int32_t v1, int32_t v2, int32_t v3, DType* orientedArea) {
        DType vec1[3]; // = vs4[order[i][1]] - vs4[order[i][0]]; // HalfEdgeVec(pHE1);
        vec3Minus(allVertsArray + PointVecDims * v2, allVertsArray + PointVecDims * v1, vec1);

        DType vec2[3]; // = vs4[order[i][2]] - vs4[order[i][1]];  // HalfEdgeVec(pHE2);
        vec3Minus(allVertsArray + PointVecDims * v3, allVertsArray + PointVecDims * v2, vec2);
        vec3CrossProduct(vec1, vec2, orientedArea);
    }

    // from Building an Orthonormal Basis, Revisited
    template<typename DType>
    void buildOrthonormalBasis(const DType* n, DType* b1, DType* b2)
    {
        if (n[2] < 0.) {
            const float a = 1.0f / (1.0f - n[2]);
            const float b = n[0] * n[1] * a;
            b1[0] = 1.0f - n[0] * n[0] * a;
            b1[1] = -b;
            b1[2] = n[0];

            b2[0] = b;
            b2[1] = n[1] * n[1] * a - 1.0f;
            b2[2] = -n[1];
        }
        else {
            const float a = 1.0f / (1.0f + n[2]);
            const float b = -n[0] * n[1] * a;
            b1[0] = 1.0f - n[0] * n[0] * a;
            b1[1] = b;
            b1[2] = -n[0];

            b2[0] = b;
            b2[1] = 1.0f - n[1] * n[1] * a;
            b2[2] = -n[1];
        }
    }


};