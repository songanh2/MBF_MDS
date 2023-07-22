from pyspark.sql import SparkSession
from functools import reduce
import sys,datetime

def main():      
    try:          
        #output_dir=sys.argv[1]
        #day_key=sys.argv[2]
        #hour_key=sys.argv[3]
        #print(output_dir)
        #print(day_key)
        #print(hour_key)        
        #mo_key='202211'
        mo_key=sys.argv[1]   
        output_dir='/DATALAKE_TLS_TEST/EXP_CHURN_POSTPAID_PREDICT_DATA/' + mo_key      
        # Init spark session 
        sparkSession = SparkSession.builder.appName("EXP_CHURN_POSTPAID_PREDICT_DATA").config("spark.sql.crossJoin.enabled", "true").enableHiveSupport().getOrCreate();
        
        # Step - 1
        # Xac dinh tap thue bao CHURN_IND = 1(Thue bao tra sau roi mang)
        df = sparkSession.sql("""  SELECT 
                                                A.SERVICE_NBR                                                                               AS SERVICE_NBR,                                                 
                                                A.ACCT_SRVC_INSTANCE_KEY                                                                    AS ACCT_SRVC_INSTANCE_KEY,
                                                A.ACCT_KEY                                                                                  AS ACCT_KEY,
                                                A.CUST_KEY                                                                                  AS CUST_KEY,
                                                A.PROD_LINE_KEY                                                                             AS PROD_LINE_KEY,
                                                A.BSNS_RGN_KEY	 																			                                      AS BSNS_RGN_KEY,
                                                A.GEO_STATE_CD 																				                                      AS GEO_STATE_CD,
                                                A.GEO_DSTRCT_CD 																			                                      AS GEO_DSTRCT_CD,
                                                A.GEO_CITY_CD 																				                                      AS GEO_CITY_CD,
                                                INACTIVITY_DAYS_CNT                                                                         AS INACTIVITY_DAYS_CNT,
                                                MAIN_ACCT_BAL_MO                                                                            AS MAIN_ACCT_BAL_MO,
                                                MAIN_ACCT_BAL_2MO                                                                           AS MAIN_ACCT_BAL_2MO, 
                                                MAIN_ACCT_BAL_3MO                                                                           AS MAIN_ACCT_BAL_3MO,  
                                                (RGB_DAYS_CNT_MO + RGB_DAYS_CNT_2MO + RGB_DAYS_CNT_3MO)                                     AS RGB_DAYS_CNT, 
                                                (DATA_RGB_DAYS_CNT_MO + DATA_RGB_DAYS_CNT_2MO + DATA_RGB_DAYS_CNT_3MO)                      AS DATA_RGB_DAYS_CNT, 
                                                RVN_AMT_MO                                                                                  AS RVN_AMT_MO,
                                                RVN_AMT_2MO                                                                                 AS RVN_AMT_2MO,
                                                RVN_AMT_3MO                                                                                 AS RVN_AMT_3MO,                                                
                                                RCHRG_AMT_MO                                                                                AS RCHRG_AMT_MO,
                                                RCHRG_AMT_2MO                                                                               AS RCHRG_AMT_2MO,
                                                RCHRG_AMT_3MO                                                                               AS RCHRG_AMT_3MO,                                                   
                                                VOICE_USG_MO                                                                                AS VOICE_USG_MO,
                                                VOICE_USG_2MO                                                                               AS VOICE_USG_2MO,
                                                VOICE_USG_3MO                                                                               AS VOICE_USG_3MO,            
                                                (VOICE_ONNET_USG_MO + VOICE_ONNET_USG_2MO + VOICE_ONNET_USG_3MO)                            AS VOICE_ONNET_USG, 
                                                (VOICE_OFFNET_USG_MO + VOICE_OFFNET_USG_2MO + VOICE_OFFNET_USG_3MO)                         AS VOICE_OFFNET_USG,
                                                DATA_USG_MO                                                                                 AS DATA_USG_MO,
                                                DATA_USG_2MO                                                                                AS DATA_USG_2MO,
                                                DATA_USG_3MO                                                                                AS DATA_USG_3MO,                                                
                                                VLR_ATTACHED_DAYS_CNT_MO                                                                    AS VLR_ATTACHED_DAYS_CNT_MO,
                                                VLR_ATTACHED_DAYS_CNT_2MO                                                                   AS VLR_ATTACHED_DAYS_CNT_2MO,
                                                VLR_ATTACHED_DAYS_CNT_3MO                                                                   AS VLR_ATTACHED_DAYS_CNT_3MO,
                                                (RMNG_DAYS_CNT_MO + RMNG_DAYS_CNT_2MO + RMNG_DAYS_CNT_3MO)                                  AS RMNG_DAYS_CNT,                                               
                                                (VN_CALL_NUM_MO + VN_CALL_NUM_2MO + VN_CALL_NUM_3MO)                                        AS VN_CALL_NUM,                                                
                                                (VN_DURATION_MO + VN_DURATION_2MO + VN_DURATION_3MO)                                        AS VN_DURATION,
                                                (VN_CALL_FAIL_MO + VN_CALL_FAIL_2MO + VN_CALL_FAIL_3MO)                                     AS VN_CALL_FAIL,                                                
                                                (VN_CALL_DROP_MO + VN_CALL_DROP_2MO + VN_CALL_DROP_3MO)                                     AS VN_CALL_DROP,                   
                                                (VN_CSSR_MO + VN_CSSR_2MO + VN_CSSR_3MO)                                                    AS VN_CSSR,
                                                (VN_CDR_MO + VN_CDR_2MO + VN_CDR_3MO)                                                       AS VN_CDR,
                                                (VN_CST_MO + VN_CST_2MO + VN_CST_3MO)                                                       AS VN_CST,
                                                (W_RES_DL_MO + W_RES_DL_2MO + W_RES_DL_3MO)                                                 AS W_RES_DL,
                                                (W_RES_SR_MO + W_RES_SR_2MO + W_RES_SR_3MO)                                                 AS W_RES_SR,
                                                (W_BRO_SR_MO + W_BRO_SR_2MO + W_BRO_SR_3MO)                                                 AS W_BRO_SR,
                                                (W_DPL_DL_MO + W_DPL_DL_2MO + W_DPL_DL_3MO)                                                 AS W_DPL_DL,
                                                (W_DL_THP_MO + W_DL_THP_2MO + W_DL_THP_3MO)                                                 AS W_DL_THP,
                                                (V_STR_SR_MO + V_STR_SR_2MO + V_STR_SR_3MO)                                                 AS V_STR_SR,
                                                (V_STR_DL_MO + V_STR_DL_2MO + V_STR_DL_3MO)                                                 AS V_STR_DL,
                                                (V_STA_RT_MO + V_STA_RT_2MO + V_STA_RT_3MO)                                                 AS V_STA_RT,
                                                (V_DL_THP_MO + V_DL_THP_2MO + V_DL_THP_3MO)                                                 AS V_DL_THP,
                                                (CALL_MCA_MO + CALL_MCA_2MO + CALL_MCA_3MO)                                                 AS CALL_MCA, 
                                                INVC_AMT_MO                                                                                 AS INVC_AMT_MO,
                                                INVC_AMT_2MO                                                                                AS INVC_AMT_2MO,
                                                INVC_AMT_3MO                                                                                AS INVC_AMT_3MO,                                                                           
                                                SATISFACTION_LEVEL                                                                          AS SATISFACTION_LEVEL,
                                                CHURN_IND                                                                                   AS CHURN_IND 
                                            FROM MBF_BIGDATA.ADMR_ACCOUNT_SERVICE A 
                                            JOIN MBF_BIGDATA.ADMD_CHURN_KPI_MO B ON A.SERVICE_NBR = B.SERVICE_NBR AND A.ACCT_SRVC_INSTANCE_KEY = B.ACCT_SRVC_INSTANCE_KEY AND B.MO_KEY = '202207' AND CHURN_IND = 0
                                            WHERE A.DAY_KEY = '20220731'  
                                            AND A.PROD_LINE_KEY = 2  
                                            """)                          
                                            #AND A.NTWK_QOS_GRP_CD NOT IN('FC')
                                            #AND A.STAT_CD NOT IN('02','22','20','12','21')                                                                                   

        print('Tong so mau dua vao mining: ' + str(df.count()))        
        df.coalesce(1).write.mode("overwrite").option('header','true').csv(output_dir)        
    except Exception as err:
        raise
    finally:      
        #Spark stop  
        if(sparkSession != None):
            sparkSession.stop()                
            
if __name__ == '__main__':
    main()