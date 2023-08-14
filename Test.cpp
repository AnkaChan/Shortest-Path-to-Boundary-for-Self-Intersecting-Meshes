#include "ShortestPath/CollisionDetector/CollisionDetertionParameters.h"
#include "ShortestPath/CollisionDetector/DiscreteCollisionDetector.h"
#include "ShortestPath/TetMesh/TetMeshFEM.h"

using namespace SP;

int main(int argc, char ** argv) 
{
	if (argc < 3)
	{
		std::cout << "Please provide 2 arguments: [Collision Detector Setting File] [Input Tet Mesh]" << std::endl;
		return -1;
	}

	CollisionDetectionParamters params;
	params.loadFromJsonFile(argv[1]);

	DiscreteCollisionDetector dcd(params);

	// MeshFrame mesh used to load the mesh and precompute the topology
	TetMeshMF::SharedPtr pMeshMF = std::make_shared<TetMeshMF>();
	pMeshMF->load_t(argv[2]);

	// Tetmesh used by the shortest path algorithn
	TetMeshFEM::SharedPtr pMesh = std::make_shared<TetMeshFEM>();
	// configure the physics parameters of the mesh, isn't really used for the shortest path algorithm
	ObjectParams::SharedPtr pObjParams = std::make_shared<ObjectParams>();
	pMesh->initialize(pObjParams, pMeshMF);

	dcd.initialize({ pMesh });

	// after updating the mesh you need to update the BVH
	pMesh->vertex(0)(0) += 0.01;
	// if the rebuild quality is set to refit then the BVH will not be reconstructed
	dcd.updateBVH(RTC_BUILD_QUALITY_REFIT, RTC_BUILD_QUALITY_REFIT, true);

	// collision detection and shortest path find
	int meshId = 0;
	for (size_t iV = 0; iV < pMesh->numVertices(); iV++)
	{
		CollisionDetectionResult colDecResult;
		dcd.vertexCollisionDetection(iV, meshId, &colDecResult);
		ClosestPointQueryResult queryResult;
		dcd.closestPointQuery(&colDecResult, &queryResult);

		if (colDecResult.numIntersections())
		{
			std::cout << "-----------------------------\n";
			std::cout << "Intersection found for vertex: " << iV   
				<< ". Intersection times: " << colDecResult.numIntersections() << "\n";
			for (size_t iIntersection = 0; iIntersection < colDecResult.numIntersections(); iIntersection++)
			{
				std::cout << "    Intersection: " << iIntersection
					<< " | intersecting tet: " << colDecResult.intersectedTets[iIntersection]
					<< " | closest point: " << colDecResult.closestSurfacePts[iIntersection][0] << " "
					<< colDecResult.closestSurfacePts[iIntersection][1] << " "
					<< colDecResult.closestSurfacePts[iIntersection][2]
					<< " | from surface face: " << colDecResult.closestSurfaceFaceId[iIntersection]
					<< "\n";
			}
		}
	}

}