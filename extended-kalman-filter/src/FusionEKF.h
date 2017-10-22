#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"


class FusionEKF {
public:
  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

private:

  // Constant value for minimum state value.
  static constexpr float kMinStateValue = 0.0001;
  // Constant value for acceleration noise in x and y.
  static constexpr float kNoiseAx = 9;
  static constexpr float kNoiseAy = 9;

  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;

  // Computes the Process Covariance Matrix.
  void ComputeProcessCovarianceMatrix(float dt);

  // Initializes the KalmanFilter state using the first measurement.
  void Initialize(const MeasurementPackage &measurement_pack);

  // Ensures the state is stable by avoiding values too close to 0.
  void MaybeCorrectState();

  // Updates the state and the covariance matrix.
  void Update(const MeasurementPackage &measurement_pack);
};

#endif /* FusionEKF_H_ */
