#pragma once

#include "MeshFrame/Utility/Parser.h"
#include <MeshFrame/Memory/Array.h>
#define PREALLOCATED_NUM_COLLISIONS 1

namespace SP {

    struct CollisionStatistics : public MF::BaseJsonConfig
    {
        // number of total sub-timesteps
        std::vector<int> numOfCollisionsDCDs;
        std::vector<int> numOfCollisionsCCDs;
        // number of total sub-timesteps x number of DCD result
        std::vector<std::vector<int>> numOfBVHQuerysEachStep;
        std::vector<std::vector<int>> numOfTetTraversal;
        std::vector<std::vector<int>> numTetsTraversed;

        virtual bool fromJson(nlohmann::json& collisionParam) {
            EXTRACT_FROM_JSON(collisionParam, numOfCollisionsDCDs);
            EXTRACT_FROM_JSON(collisionParam, numOfCollisionsCCDs);
            EXTRACT_FROM_JSON(collisionParam, numOfBVHQuerysEachStep);
            EXTRACT_FROM_JSON(collisionParam, numOfTetTraversal);
            EXTRACT_FROM_JSON(collisionParam, numTetsTraversed);

            return true;
        }

        virtual bool toJson(nlohmann::json& collisionParam) {
            PUT_TO_JSON(collisionParam, numOfCollisionsDCDs);
            PUT_TO_JSON(collisionParam, numOfCollisionsCCDs);
            PUT_TO_JSON(collisionParam, numOfBVHQuerysEachStep);
            PUT_TO_JSON(collisionParam, numOfTetTraversal);
            PUT_TO_JSON(collisionParam, numTetsTraversed);

            return true;

        }
    };

	struct CollisionDetectionParamters : public MF::BaseJsonConfig
	{
        // collision detection parameters
        bool allowCCD = true;
        bool allowDCD = true;

        bool checkFeasibleRegion = true;
        bool checkTetTraverse = true;
        bool handleSelfCollision = true;
        bool stopTraversingAfterPassingQueryPoint = true;
        bool tetrahedralTraverseForNonSelfIntersection = true;
        bool useStaticTraverse = true;
        bool restPoseCloestPoint = false;
        bool loopLessTraverse = false;

        bool shiftQueryPointToCenter = true;
        float centerShiftLevel = 0.01f;
        int maxNumberOfBVHQuery = 500;

        int numberOfBVHQuery = 0;
        int numberOfTetTraversal = 0;
        int numberOfTetsTraversed = 0;

        // tetrahedral traverse parameters
        float feasibleRegionEpsilon = 1e-2f;
        float rayTriIntersectionEPSILON = 1e-10f;
        float maxSearchDistanceMultiplier = 1.8f;

        virtual bool fromJson(nlohmann::json& collisionParam) {
            EXTRACT_FROM_JSON(collisionParam, allowCCD);
            EXTRACT_FROM_JSON(collisionParam, allowDCD);
            EXTRACT_FROM_JSON(collisionParam, checkTetTraverse);
            EXTRACT_FROM_JSON(collisionParam, checkFeasibleRegion);
            EXTRACT_FROM_JSON(collisionParam, tetrahedralTraverseForNonSelfIntersection);
            EXTRACT_FROM_JSON(collisionParam, handleSelfCollision);
            EXTRACT_FROM_JSON(collisionParam, stopTraversingAfterPassingQueryPoint);

            EXTRACT_FROM_JSON(collisionParam, shiftQueryPointToCenter);
            EXTRACT_FROM_JSON(collisionParam, centerShiftLevel);

            EXTRACT_FROM_JSON(collisionParam, feasibleRegionEpsilon);
            EXTRACT_FROM_JSON(collisionParam, rayTriIntersectionEPSILON);
            EXTRACT_FROM_JSON(collisionParam, maxSearchDistanceMultiplier);
            EXTRACT_FROM_JSON(collisionParam, useStaticTraverse);

            EXTRACT_FROM_JSON(collisionParam, restPoseCloestPoint);
            EXTRACT_FROM_JSON(collisionParam, loopLessTraverse);



            return true;
        }

        virtual bool toJson(nlohmann::json& collisionParam) {
            PUT_TO_JSON(collisionParam, allowCCD);
            PUT_TO_JSON(collisionParam, allowDCD);
            PUT_TO_JSON(collisionParam, checkTetTraverse);
            PUT_TO_JSON(collisionParam, checkFeasibleRegion);
            PUT_TO_JSON(collisionParam, tetrahedralTraverseForNonSelfIntersection);
            PUT_TO_JSON(collisionParam, handleSelfCollision);
            PUT_TO_JSON(collisionParam, stopTraversingAfterPassingQueryPoint);

            PUT_TO_JSON(collisionParam, shiftQueryPointToCenter);
            PUT_TO_JSON(collisionParam, centerShiftLevel);

            PUT_TO_JSON(collisionParam, feasibleRegionEpsilon);
            PUT_TO_JSON(collisionParam, rayTriIntersectionEPSILON);
            PUT_TO_JSON(collisionParam, maxSearchDistanceMultiplier);
            PUT_TO_JSON(collisionParam, useStaticTraverse);

            PUT_TO_JSON(collisionParam, restPoseCloestPoint);
            PUT_TO_JSON(collisionParam, loopLessTraverse);


            return true;

        }
	};

    enum class ClosestPointOnTriangleType
    {
        AtA,
        AtB,
        AtC,
        AtAB,
        AtBC,
        AtAC,
        AtInterior,
        NotFound
    };

    struct DiscreteCollisionDetector;

    struct CollisionDetectionResult
    {
        CollisionDetectionResult()
        {}

        int numIntersections() { return intersectedTMeshIds.size(); }
        void clear() { 
            shortestPathFound.clear();
            intersectedTets.clear();
            intersectedTMeshIds.clear();
            closestSurfaceFaceId.clear();
            closestSurfacePts.clear();
            closestSurfacePtBarycentrics.clear();
            closestPointType.clear();

            numberOfBVHQuery = 0;
            numberOfTetTraversal = 0;
            numberOfTetsTraversed = 0;
        }

        CPArray<bool, PREALLOCATED_NUM_COLLISIONS> shortestPathFound;
        // set to non-negative when doing vertex collision detection
        int idVQuery = -1;
        // set to non-nullptr when doing tet centroid collision detection
        int idTetQuery = -1;

        int idTMQuery = -1;
        // if pTMToCheck is set to non-null it will only detect collision between pVQuery and pTMToCheck
        // int pTMToCheck = nullptr;

        // for DCD only
        CPArray<int, PREALLOCATED_NUM_COLLISIONS> intersectedTets;
        // for DCD + CCD
        CPArray<int, PREALLOCATED_NUM_COLLISIONS> intersectedTMeshIds;

        // collision solving informations
        // for DCD + CCD
        CPArray<int, PREALLOCATED_NUM_COLLISIONS> closestSurfaceFaceId;
        // for DCD only
        CPArray<std::array<float, 3>, PREALLOCATED_NUM_COLLISIONS> closestSurfacePts;
        // for DCD only
        CPArray<std::array<float, 3>, PREALLOCATED_NUM_COLLISIONS> closestSurfacePtBarycentrics;
        // for DCD only, thus it need to be recomputed for CCD results at the collision solving stage
        CPArray<ClosestPointOnTriangleType, PREALLOCATED_NUM_COLLISIONS> closestPointType;
        CPArray<std::array<float, 3>, PREALLOCATED_NUM_COLLISIONS> closestPointNormals;

        //std::map<unsigned int, PathFinder::TM::Ptr>* pTetmeshGeoIdToPointerMap;
        //std::map<PathFinder::TM::Ptr, unsigned int>* pTetmeshPtrToTetMeshIndexMap;
        //std::vector<std::vector<PathFinder::TM::TPtr>>* pTetTraversed = nullptr;
        //std::vector<PathFinder::TM::Ptr>* pTetMeshPtrs;
        bool handleSelfIntersection = true;
        bool fromCCD = false;

        float penetrationDepth = -1.f;

        int numberOfBVHQuery = 0;
        int numberOfTetTraversal = 0;
        int numberOfTetsTraversed = 0;

        // either CCD or DCD
        void* pDetector = nullptr;
    };
}